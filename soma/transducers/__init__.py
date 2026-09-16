"""R2 transducer SDK: adapters translate domains to events, never solve tasks."""

from .history import BitHistoryTransducer
from .sdk import Transducer, TransducerSpec
from .bit_scalar import BitScalarTransducer
from .cart import CartTransducer
from .scalar_stream import ScalarStreamTransducer
from .symbol_bits import SymbolBitsTransducer
from .text_bytes import TextBytesTransducer, decode_bits, encode_bytes, validate_utf8
from .vision import GlyphTransducer, render_horizontal, render_vertical

__all__ = [
    "BitScalarTransducer",
    "CartTransducer",
    "GlyphTransducer",
    "ScalarStreamTransducer",
    "SymbolBitsTransducer",
    "TextBytesTransducer",
    "Transducer",
    "TransducerSpec",
    "decode_bits",
    "encode_bytes",
    "render_horizontal",
    "render_vertical",
    "validate_utf8",
]
