"""Arithmetic coder."""

import io,math,random
import typing as tp
import torch

from binary import BitPacker
# 將 PDF --> CDF
def build_stable_quantized_cdf(pdf:torch.Tensor,total_range_bits:int,
                               roundoff:float = 1e-8,min_range:int=2,
                               check:bool=True)->torch.Tensor:
    pdf=pdf.detach()
    # roundoff 為進行小數點後幾位的四捨五入，以利浮點數的的精度
    if roundoff:
        pdf = (pdf / roundoff).floor() * roundoff
        
    # total_range: 轉成CDF的整體可用範圍
    total_range = 2 ** total_range_bits 
    cardinality = len(pdf)
    # alpha: 用來調整pdf
    alpha = min_range * cardinality / total_range
    assert alpha <= 1, "you must reduce total_range"
    ranges = (((1 - alpha) * total_range) * pdf).floor().long()
    # 確保ranges有介在min ~ total 之間
    ranges += min_range
    #將range轉換成quantized_CDF
    quantized_cdf = torch.cumsum(ranges, dim=-1)
    if min_range < 2:
        raise ValueError("min_range must be at least 2.")
    # check: 是否要進行錯誤檢查
    if check:
        assert quantized_cdf[-1] <= 2 ** total_range_bits, quantized_cdf[-1]
        #確保有在total_range的範圍內
        if ((quantized_cdf[1:] - quantized_cdf[:-1]) < min_range).any() or quantized_cdf[0] < min_range:
            raise ValueError("You must increase your total_range_bits.")
    return quantized_cdf   


# 建立 ArithmeticCoder
class ArithmeticCoder:
   
    def __init__(self, fo: tp.IO[bytes], total_range_bits: int = 24):
        assert total_range_bits <= 30
        self.total_range_bits = total_range_bits
        self.packer = BitPacker(bits=1, fo=fo)  # 資訊包成一個文件
        self.low: int = 0                 # 範圍的上限
        self.high: int = 0                # 範圍的下限
        self.max_bit: int = -1            # 範圍內的最高位數
        self._dbg: tp.List[tp.Any] = []   # debug 用
        self._dbg2: tp.List[tp.Any] = []  # 同上

    @property
    def delta(self) -> int:
        """Return the current range width."""
        return self.high - self.low + 1

    def _flush_common_prefix(self):
        # If self.low and self.high start with the sames bits,
        # those won't change anymore as we always just increase the range
        # by powers of 2, and we can flush them out to the bit stream.
        assert self.high >= self.low, (self.low, self.high)
        assert self.high < 2 ** (self.max_bit + 1)
        while self.max_bit >= 0:
            b1 = self.low >> self.max_bit
            b2 = self.high >> self.max_bit
            if b1 == b2:
                self.low -= (b1 << self.max_bit)
                self.high -= (b1 << self.max_bit)
                assert self.high >= self.low, (self.high, self.low, self.max_bit)
                assert self.low >= 0
                self.max_bit -= 1
                self.packer.push(b1)
            else:
                break

    def push(self, symbol: int, quantized_cdf: torch.Tensor):
        """Push the given symbol on the stream, flushing out bits
        if possible.

        Args:
            symbol (int): symbol to encode with the AC.
            quantized_cdf (torch.Tensor): use `build_stable_quantized_cdf`
                to build this from your pdf estimate.
        """
        while self.delta < 2 ** self.total_range_bits:
            self.low *= 2
            self.high = self.high * 2 + 1
            self.max_bit += 1

        range_low = 0 if symbol == 0 else quantized_cdf[symbol - 1].item()
        range_high = quantized_cdf[symbol].item() - 1
        effective_low = int(math.ceil(range_low * (self.delta / (2 ** self.total_range_bits))))
        effective_high = int(math.floor(range_high * (self.delta / (2 ** self.total_range_bits))))
        assert self.low <= self.high
        self.high = self.low + effective_high
        self.low = self.low + effective_low
        assert self.low <= self.high, (effective_low, effective_high, range_low, range_high)
        self._dbg.append((self.low, self.high))
        self._dbg2.append((self.low, self.high))
        outs = self._flush_common_prefix()
        assert self.low <= self.high
        assert self.max_bit >= -1
        assert self.max_bit <= 61, self.max_bit
        return outs

    def flush(self):
        """Flush the remaining information to the stream.
        """
        while self.max_bit >= 0:
            b1 = (self.low >> self.max_bit) & 1
            self.packer.push(b1)
            self.max_bit -= 1
        self.packer.flush()
        
        
class ArithmeticDecoder:
    """ArithmeticDecoder, see `ArithmeticCoder` for a detailed explanation.

    Note that this must be called with **exactly** the same parameters and sequence
    of quantized cdf as the arithmetic encoder or the wrong values will be decoded.

    If the AC encoder current range is [L, H], with `L` and `H` having the some common
    prefix (i.e. the same most significant bits), then this prefix will be flushed to the stream.
    For instances, having read 3 bits `b1 b2 b3`, we know that `[L, H]` is contained inside
    `[b1 b2 b3 0 ... 0 b1 b3 b3 1 ... 1]`. Now this specific sub-range can only be obtained
    for a specific sequence of symbols and a binary-search allows us to decode those symbols.
    At some point, the prefix `b1 b2 b3` will no longer be sufficient to decode new symbols,
    and we will need to read new bits from the stream and repeat the process.

    """
    def __init__(self, fo: tp.IO[bytes], total_range_bits: int = 24):
        self.total_range_bits = total_range_bits
        self.low: int = 0
        self.high: int = 0
        self.current: int = 0
        self.max_bit: int = -1
        self.unpacker = BitUnpacker(bits=1, fo=fo)  # we pull single bits at a time.
        # Following is for debugging
        self._dbg: tp.List[tp.Any] = []
        self._dbg2: tp.List[tp.Any] = []
        self._last: tp.Any = None

    @property
    def delta(self) -> int:
        return self.high - self.low + 1

    def _flush_common_prefix(self):
        # Given the current range [L, H], if both have a common prefix,
        # we know we can remove it from our representation to avoid handling large numbers.
        while self.max_bit >= 0:
            b1 = self.low >> self.max_bit
            b2 = self.high >> self.max_bit
            if b1 == b2:
                self.low -= (b1 << self.max_bit)
                self.high -= (b1 << self.max_bit)
                self.current -= (b1 << self.max_bit)
                assert self.high >= self.low
                assert self.low >= 0
                self.max_bit -= 1
            else:
                break

    def pull(self, quantized_cdf: torch.Tensor) -> tp.Optional[int]:
        """Pull a symbol, reading as many bits from the stream as required.
        This returns `None` when the stream has been exhausted.

        Args:
            quantized_cdf (torch.Tensor): use `build_stable_quantized_cdf`
                to build this from your pdf estimate. This must be **exatly**
                the same cdf as the one used at encoding time.
        """
        while self.delta < 2 ** self.total_range_bits:
            bit = self.unpacker.pull()
            if bit is None:
                return None
            self.low *= 2
            self.high = self.high * 2 + 1
            self.current = self.current * 2 + bit
            self.max_bit += 1

        def bin_search(low_idx: int, high_idx: int):
            # Binary search is not just for coding interviews :)
            if high_idx < low_idx:
                raise RuntimeError("Binary search failed")
            mid = (low_idx + high_idx) // 2
            range_low = quantized_cdf[mid - 1].item() if mid > 0 else 0
            range_high = quantized_cdf[mid].item() - 1
            effective_low = int(math.ceil(range_low * (self.delta / (2 ** self.total_range_bits))))
            effective_high = int(math.floor(range_high * (self.delta / (2 ** self.total_range_bits))))
            low = effective_low + self.low
            high = effective_high + self.low
            if self.current >= low:
                if self.current <= high:
                    return (mid, low, high, self.current)
                else:
                    return bin_search(mid + 1, high_idx)
            else:
                return bin_search(low_idx, mid - 1)

        self._last = (self.low, self.high, self.current, self.max_bit)
        sym, self.low, self.high, self.current = bin_search(0, len(quantized_cdf) - 1)
        self._dbg.append((self.low, self.high, self.current))
        self._flush_common_prefix()
        self._dbg2.append((self.low, self.high, self.current))

        return sym

# 測試AC是否有問題
def test():
    torch.manual_seed(1234)
    random.seed(1234)
    for _ in range(4):
        pdfs = []
        cardinality = random.randrange(4000)
        steps = random.randrange(100, 500)
        fo = io.BytesIO()
        encoder = ArithmeticCoder(fo)
        symbols = []
        for step in range(steps):
            pdf = torch.softmax(torch.randn(cardinality), dim=0)
            pdfs.append(pdf)
            q_cdf = build_stable_quantized_cdf(pdf, encoder.total_range_bits)
            symbol = torch.multinomial(pdf, 1).item()
            symbols.append(symbol)
            encoder.push(symbol, q_cdf)
        encoder.flush()

        fo.seek(0)
        decoder = ArithmeticDecoder(fo)
        for idx, (pdf, symbol) in enumerate(zip(pdfs, symbols)):
            q_cdf = build_stable_quantized_cdf(pdf, encoder.total_range_bits)
            decoded_symbol = decoder.pull(q_cdf)
            assert decoded_symbol == symbol, idx
        assert decoder.pull(torch.zeros(1)) is None


if __name__ == "__main__":
    test()