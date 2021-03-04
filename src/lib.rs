// Ported from <https://github.com/python/cpython/blob/master/Python/marshal.c>
use bitflags::bitflags;
use num_bigint::BigInt;
use num_complex::Complex;
use num_derive::{FromPrimitive, ToPrimitive};
use std::{
    collections::{HashMap, HashSet},
    convert::TryFrom,
    fmt,
    hash::{Hash, Hasher},
    iter::FromIterator,
    sync::{Arc, RwLock},
};

/// `Arc` = immutable
/// `ArcRwLock` = mutable
pub type ArcRwLock<T> = Arc<RwLock<T>>;

#[derive(FromPrimitive, ToPrimitive, Debug, Copy, Clone)]
#[repr(u8)]
enum Type {
    Null               = b'0',
    None               = b'N',
    False              = b'F',
    True               = b'T',
    StopIter           = b'S',
    Ellipsis           = b'.',
    Int                = b'i',
    Int64              = b'I',
    Float              = b'f',
    BinaryFloat        = b'g',
    Complex            = b'x',
    BinaryComplex      = b'y',
    Long               = b'l',
    String             = b's',
    Interned           = b't',
    Ref                = b'r',
    Tuple              = b'(',
    List               = b'[',
    Dict               = b'{',
    Code               = b'c',
    Unicode            = b'u',
    Unknown            = b'?',
    Set                = b'<',
    FrozenSet          = b'>',
    Ascii              = b'a',
    AsciiInterned      = b'A',
    SmallTuple         = b')',
    ShortAscii         = b'z',
    ShortAsciiInterned = b'Z',
}
impl Type {
    const FLAG_REF: u8 = b'\x80';
}

struct Depth(Arc<()>);
impl Depth {
    const MAX: usize = 900;

    #[must_use]
    pub fn new() -> Self {
        Self(Arc::new(()))
    }

    pub fn try_clone(&self) -> Option<Self> {
        if Arc::strong_count(&self.0) > Self::MAX {
            None
        } else {
            Some(Self(self.0.clone()))
        }
    }
}
impl fmt::Debug for Depth {
    fn fmt(&self, f: &mut fmt::Formatter) -> Result<(), fmt::Error> {
        f.debug_tuple("Depth")
            .field(&Arc::strong_count(&self.0))
            .finish()
    }
}

bitflags! {
    pub struct CodeFlags: u32 {
        const OPTIMIZED                   = 0x1;
        const NEWLOCALS                   = 0x2;
        const VARARGS                     = 0x4;
        const VARKEYWORDS                 = 0x8;
        const NESTED                     = 0x10;
        const GENERATOR                  = 0x20;
        const NOFREE                     = 0x40;
        const COROUTINE                  = 0x80;
        const ITERABLE_COROUTINE        = 0x100;
        const ASYNC_GENERATOR           = 0x200;
        // TODO: old versions
        const GENERATOR_ALLOWED        = 0x1000;
        const FUTURE_DIVISION          = 0x2000;
        const FUTURE_ABSOLUTE_IMPORT   = 0x4000;
        const FUTURE_WITH_STATEMENT    = 0x8000;
        const FUTURE_PRINT_FUNCTION   = 0x10000;
        const FUTURE_UNICODE_LITERALS = 0x20000;
        const FUTURE_BARRY_AS_BDFL    = 0x40000;
        const FUTURE_GENERATOR_STOP   = 0x80000;
        #[allow(clippy::unreadable_literal)]
        const FUTURE_ANNOTATIONS     = 0x100000;
    }
}

#[rustfmt::skip]
#[derive(Clone, Debug)]
pub struct Code {
    pub argcount:        u32,
    pub posonlyargcount: u32,
    pub kwonlyargcount:  u32,
    pub nlocals:         u32,
    pub stacksize:       u32,
    pub flags:           CodeFlags,
    pub code:            Arc<Vec<u8>>,
    pub consts:          Arc<Vec<Obj>>,
    pub names:           Vec<Arc<String>>,
    pub varnames:        Vec<Arc<String>>,
    pub freevars:        Vec<Arc<String>>,
    pub cellvars:        Vec<Arc<String>>,
    pub filename:        Arc<String>,
    pub name:            Arc<String>,
    pub firstlineno:     u32,
    pub lnotab:          Arc<Vec<u8>>,
}

#[rustfmt::skip]
#[derive(Clone)]
pub enum Obj {
    None,
    StopIteration,
    Ellipsis,
    Bool     (bool),
    Long     (Arc<BigInt>),
    Float    (f64),
    Complex  (Complex<f64>),
    Bytes    (Arc<Vec<u8>>),
    String   (Arc<String>),
    Tuple    (Arc<Vec<Obj>>),
    List     (ArcRwLock<Vec<Obj>>),
    Dict     (ArcRwLock<HashMap<ObjHashable, Obj>>),
    Set      (ArcRwLock<HashSet<ObjHashable>>),
    FrozenSet(Arc<HashSet<ObjHashable>>),
    Code     (Arc<Code>),
    // etc.
}
macro_rules! define_extract {
    ($extract_fn:ident($variant:ident) -> ()) => {
        define_extract! { $extract_fn -> () { $variant => () } }
    };
    ($extract_fn:ident($variant:ident) -> Arc<$ret:ty>) => {
        define_extract! { $extract_fn -> Arc<$ret> { $variant(x) => x } }
    };
    ($extract_fn:ident($variant:ident) -> ArcRwLock<$ret:ty>) => {
        define_extract! { $extract_fn -> ArcRwLock<$ret> { $variant(x) => x } }
    };
    ($extract_fn:ident($variant:ident) -> $ret:ty) => {
        define_extract! { $extract_fn -> $ret { $variant(x) => x } }
    };
    ($extract_fn:ident -> $ret:ty { $variant:ident$(($($pat:pat),+))? => $expr:expr }) => {
        /// # Errors
        /// Returns a reference to self if extraction fails
        pub fn $extract_fn(self) -> Result<$ret, Self> {
            if let Self::$variant$(($($pat),+))? = self {
                Ok($expr)
            } else {
                Err(self)
            }
        }
    }
}
macro_rules! define_is {
    ($is_fn:ident($variant:ident$(($($pat:pat),+))?)) => {
        /// # Errors
        /// Returns a reference to self if extraction fails
        #[must_use]
        pub fn $is_fn(&self) -> bool {
            if let Self::$variant$(($($pat),+))? = self {
                true
            } else {
                false
            }
        }
    }
}
impl Obj {
    define_extract! { extract_none          (None)          -> ()                                    }
    define_extract! { extract_stop_iteration(StopIteration) -> ()                                    }
    define_extract! { extract_bool          (Bool)          -> bool                                  }
    define_extract! { extract_long          (Long)          -> Arc<BigInt>                           }
    define_extract! { extract_float         (Float)         -> f64                                   }
    define_extract! { extract_bytes         (Bytes)         -> Arc<Vec<u8>>                          }
    define_extract! { extract_string        (String)        -> Arc<String>                           }
    define_extract! { extract_tuple         (Tuple)         -> Arc<Vec<Self>>                        }
    define_extract! { extract_list          (List)          -> ArcRwLock<Vec<Self>>                  }
    define_extract! { extract_dict          (Dict)          -> ArcRwLock<HashMap<ObjHashable, Self>> }
    define_extract! { extract_set           (Set)           -> ArcRwLock<HashSet<ObjHashable>>       }
    define_extract! { extract_frozenset     (FrozenSet)     -> Arc<HashSet<ObjHashable>>             }
    define_extract! { extract_code          (Code)          -> Arc<Code>                             }

    define_is! { is_none          (None)          }
    define_is! { is_stop_iteration(StopIteration) }
    define_is! { is_bool          (Bool(_))       }
    define_is! { is_long          (Long(_))       }
    define_is! { is_float         (Float(_))      }
    define_is! { is_bytes         (Bytes(_))      }
    define_is! { is_string        (String(_))     }
    define_is! { is_tuple         (Tuple(_))      }
    define_is! { is_list          (List(_))       }
    define_is! { is_dict          (Dict(_))       }
    define_is! { is_set           (Set(_))        }
    define_is! { is_frozenset     (FrozenSet(_))  }
    define_is! { is_code          (Code(_))       }
}
/// Should mostly match Python's repr
///
/// # Float, Complex
/// - Uses `float('...')` instead of `...` for nan, inf, and -inf.
/// - Uses Rust's float-to-decimal conversion.
///
/// # Bytes, String
/// - Always uses double-quotes
/// - Escapes both kinds of quotes
///
/// # Code
/// - Uses named arguments for readability
/// - lnotab is formatted as bytes(...) with a list of integers, instead of a bytes literal
impl fmt::Debug for Obj {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::None => write!(f, "None"),
            Self::StopIteration => write!(f, "StopIteration"),
            Self::Ellipsis => write!(f, "Ellipsis"),
            Self::Bool(true) => write!(f, "True"),
            Self::Bool(false) => write!(f, "False"),
            Self::Long(x) => write!(f, "{}", x),
            &Self::Float(x) => python_float_repr_full(f, x),
            &Self::Complex(x) => python_complex_repr(f, x),
            Self::Bytes(x) => python_bytes_repr(f, x),
            Self::String(x) => python_string_repr(f, x),
            Self::Tuple(x) => python_tuple_repr(f, x),
            Self::List(x) => f.debug_list().entries(x.read().unwrap().iter()).finish(),
            Self::Dict(x) => f.debug_map().entries(x.read().unwrap().iter()).finish(),
            Self::Set(x) => f.debug_set().entries(x.read().unwrap().iter()).finish(),
            Self::FrozenSet(x) => python_frozenset_repr(f, x),
            Self::Code(x) => python_code_repr(f, x),
        }
    }
}
fn python_float_repr_full(f: &mut fmt::Formatter, x: f64) -> fmt::Result {
    python_float_repr_core(f, x)?;
    if x.fract() == 0. {
        write!(f, ".0")?;
    };
    Ok(())
}
fn python_float_repr_core(f: &mut fmt::Formatter, x: f64) -> fmt::Result {
    if x.is_nan() {
        write!(f, "float('nan')")
    } else if x.is_infinite() {
        if x.is_sign_positive() {
            write!(f, "float('inf')")
        } else {
            write!(f, "-float('inf')")
        }
    } else {
        // properly handle -0.0
        if x.is_sign_negative() {
            write!(f, "-")?;
        }
        write!(f, "{}", x.abs())
    }
}
fn python_complex_repr(f: &mut fmt::Formatter, x: Complex<f64>) -> fmt::Result {
    if x.re == 0. && x.re.is_sign_positive() {
        python_float_repr_core(f, x.im)?;
        write!(f, "j")?;
    } else {
        write!(f, "(")?;
        python_float_repr_core(f, x.re)?;
        if x.im >= 0. || x.im.is_nan() {
            write!(f, "+")?;
        }
        python_float_repr_core(f, x.im)?;
        write!(f, "j)")?;
    };
    Ok(())
}
fn python_bytes_repr(f: &mut fmt::Formatter, x: &[u8]) -> fmt::Result {
    write!(f, "b\"")?;
    for &byte in x.iter() {
        match byte {
            b'\t' => write!(f, "\\t")?,
            b'\n' => write!(f, "\\n")?,
            b'\r' => write!(f, "\\r")?,
            b'\'' | b'"' | b'\\' => write!(f, "\\{}", char::from(byte))?,
            b' '..=b'~' => write!(f, "{}", char::from(byte))?,
            _ => write!(f, "\\x{:02x}", byte)?,
        }
    }
    write!(f, "\"")?;
    Ok(())
}
fn python_string_repr(f: &mut fmt::Formatter, x: &str) -> fmt::Result {
    let original = format!("{:?}", x);
    let mut last_end = 0;
    // Note: the behavior is arbitrary if there are improper escapes.
    for (start, _) in original.match_indices("\\u{") {
        f.write_str(&original[last_end..start])?;
        let len = original[start..].find('}').ok_or(fmt::Error)? + 1;
        let end = start + len;
        match len - 4 {
            0..=2 => write!(f, "\\x{:0>2}", &original[start + 3..end - 1])?,
            3..=4 => write!(f, "\\u{:0>4}", &original[start + 3..end - 1])?,
            5..=8 => write!(f, "\\U{:0>8}", &original[start + 3..end - 1])?,
            _ => panic!("Internal error: length of unicode escape = {} > 8", len),
        }
        last_end = end;
    }
    f.write_str(&original[last_end..])?;
    Ok(())
}
fn python_tuple_repr(f: &mut fmt::Formatter, x: &[Obj]) -> fmt::Result {
    if x.is_empty() {
        f.write_str("()") // Otherwise this would get formatted into an empty string
    } else {
        let mut debug_tuple = f.debug_tuple("");
        for o in x.iter() {
            debug_tuple.field(&o);
        }
        debug_tuple.finish()
    }
}
fn python_frozenset_repr(f: &mut fmt::Formatter, x: &HashSet<ObjHashable>) -> fmt::Result {
    f.write_str("frozenset(")?;
    if !x.is_empty() {
        f.debug_set().entries(x.iter()).finish()?;
    }
    f.write_str(")")?;
    Ok(())
}
fn python_code_repr(f: &mut fmt::Formatter, x: &Code) -> fmt::Result {
    write!(f, "code(argcount={:?}, posonlyargcount={:?}, kwonlyargcount={:?}, nlocals={:?}, stacksize={:?}, flags={:?}, code={:?}, consts={:?}, names={:?}, varnames={:?}, freevars={:?}, cellvars={:?}, filename={:?}, name={:?}, firstlineno={:?}, lnotab=bytes({:?}))", x.argcount, x.posonlyargcount, x.kwonlyargcount, x.nlocals, x.stacksize, x.flags, Obj::Bytes(Arc::clone(&x.code)), x.consts, x.names, x.varnames, x.freevars, x.cellvars, x.filename, x.name, x.firstlineno, &x.lnotab)
}
/// This is a f64 wrapper suitable for use as a key in a (Hash)Map, since NaNs compare equal to
/// each other, so it can implement Eq and Hash. `HashF64(-0.0) == HashF64(0.0)`.
#[derive(Clone, Debug)]
pub struct HashF64(f64);
impl PartialEq for HashF64 {
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0 || (self.0.is_nan() && other.0.is_nan())
    }
}
impl Eq for HashF64 {}
impl Hash for HashF64 {
    fn hash<H: Hasher>(&self, state: &mut H) {
        if self.0.is_nan() {
            // Multiple NaN values exist
            state.write_u8(0);
        } else if self.0 == 0.0 {
            // 0.0 == -0.0
            state.write_u8(1);
        } else {
            state.write_u64(self.0.to_bits()); // This should be fine, since all the dupes should be accounted for.
        }
    }
}

#[derive(Debug)]
pub struct HashableHashSet<T>(HashSet<T>);
impl<T> Hash for HashableHashSet<T>
where
    T: Hash,
{
    fn hash<H: Hasher>(&self, state: &mut H) {
        let mut xor: u64 = 0;
        let hasher = std::collections::hash_map::DefaultHasher::new();
        for value in &self.0 {
            let mut hasher_clone = hasher.clone();
            value.hash(&mut hasher_clone);
            xor ^= hasher_clone.finish();
        }
        state.write_u64(xor);
    }
}
impl<T> PartialEq for HashableHashSet<T>
where
    T: Eq + Hash,
{
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}
impl<T> Eq for HashableHashSet<T> where T: Eq + Hash {}
impl<T> FromIterator<T> for HashableHashSet<T>
where
    T: Eq + Hash,
{
    fn from_iter<I>(iter: I) -> Self
    where
        I: IntoIterator<Item = T>,
    {
        Self(iter.into_iter().collect())
    }
}

#[derive(PartialEq, Eq, Hash, Clone)]
pub enum ObjHashable {
    None,
    StopIteration,
    Ellipsis,
    Bool(bool),
    Long(Arc<BigInt>),
    Float(HashF64),
    Complex(Complex<HashF64>),
    String(Arc<String>),
    Tuple(Arc<Vec<ObjHashable>>),
    FrozenSet(Arc<HashableHashSet<ObjHashable>>),
    // etc.
}
impl TryFrom<&Obj> for ObjHashable {
    type Error = Obj;

    fn try_from(orig: &Obj) -> Result<Self, Obj> {
        match orig {
            Obj::None => Ok(Self::None),
            Obj::StopIteration => Ok(Self::StopIteration),
            Obj::Ellipsis => Ok(Self::Ellipsis),
            Obj::Bool(x) => Ok(Self::Bool(*x)),
            Obj::Long(x) => Ok(Self::Long(Arc::clone(x))),
            Obj::Float(x) => Ok(Self::Float(HashF64(*x))),
            Obj::Complex(Complex { re, im }) => Ok(Self::Complex(Complex {
                re: HashF64(*re),
                im: HashF64(*im),
            })),
            Obj::String(x) => Ok(Self::String(Arc::clone(x))),
            Obj::Tuple(x) => Ok(Self::Tuple(Arc::new(
                x.iter()
                    .map(Self::try_from)
                    .collect::<Result<Vec<Self>, Obj>>()?,
            ))),
            Obj::FrozenSet(x) => Ok(Self::FrozenSet(Arc::new(
                x.iter().cloned().collect::<HashableHashSet<Self>>(),
            ))),
            x => Err(x.clone()),
        }
    }
}
impl fmt::Debug for ObjHashable {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::None => write!(f, "None"),
            Self::StopIteration => write!(f, "StopIteration"),
            Self::Ellipsis => write!(f, "Ellipsis"),
            Self::Bool(true) => write!(f, "True"),
            Self::Bool(false) => write!(f, "False"),
            Self::Long(x) => write!(f, "{}", x),
            Self::Float(x) => python_float_repr_full(f, x.0),
            Self::Complex(x) => python_complex_repr(
                f,
                Complex {
                    re: x.re.0,
                    im: x.im.0,
                },
            ),
            Self::String(x) => python_string_repr(f, x),
            Self::Tuple(x) => python_tuple_hashable_repr(f, x),
            Self::FrozenSet(x) => python_frozenset_repr(f, &x.0),
        }
    }
}
fn python_tuple_hashable_repr(f: &mut fmt::Formatter, x: &[ObjHashable]) -> fmt::Result {
    if x.is_empty() {
        f.write_str("()") // Otherwise this would get formatted into an empty string
    } else {
        let mut debug_tuple = f.debug_tuple("");
        for o in x.iter() {
            debug_tuple.field(&o);
        }
        debug_tuple.finish()
    }
}

#[cfg(test)]
mod test {
    use super::{Code, CodeFlags, Obj, ObjHashable};
    use num_bigint::BigInt;
    use num_complex::Complex;
    use std::{
        collections::{HashMap, HashSet},
        sync::{Arc, RwLock},
    };

    #[test]
    fn test_debug_repr() {
        assert_eq!(format!("{:?}", Obj::None), "None");
        assert_eq!(format!("{:?}", Obj::StopIteration), "StopIteration");
        assert_eq!(format!("{:?}", Obj::Ellipsis), "Ellipsis");
        assert_eq!(format!("{:?}", Obj::Bool(true)), "True");
        assert_eq!(format!("{:?}", Obj::Bool(false)), "False");
        assert_eq!(
            format!("{:?}", Obj::Long(Arc::new(BigInt::from(-123)))),
            "-123"
        );
        assert_eq!(format!("{:?}", Obj::Tuple(Arc::new(vec![]))), "()");
        assert_eq!(
            format!("{:?}", Obj::Tuple(Arc::new(vec![Obj::Bool(true)]))),
            "(True,)"
        );
        assert_eq!(
            format!(
                "{:?}",
                Obj::Tuple(Arc::new(vec![Obj::Bool(true), Obj::None]))
            ),
            "(True, None)"
        );
        assert_eq!(
            format!(
                "{:?}",
                Obj::List(Arc::new(RwLock::new(vec![Obj::Bool(true)])))
            ),
            "[True]"
        );
        assert_eq!(
            format!(
                "{:?}",
                Obj::Dict(Arc::new(RwLock::new(
                    vec![(
                        ObjHashable::Bool(true),
                        Obj::Bytes(Arc::new(Vec::from(b"a" as &[u8])))
                    )]
                    .into_iter()
                    .collect::<HashMap<_, _>>()
                )))
            ),
            "{True: b\"a\"}"
        );
        assert_eq!(
            format!(
                "{:?}",
                Obj::Set(Arc::new(RwLock::new(
                    vec![ObjHashable::Bool(true)]
                        .into_iter()
                        .collect::<HashSet<_>>()
                )))
            ),
            "{True}"
        );
        assert_eq!(
            format!(
                "{:?}",
                Obj::FrozenSet(Arc::new(
                    vec![ObjHashable::Bool(true)]
                        .into_iter()
                        .collect::<HashSet<_>>()
                ))
            ),
            "frozenset({True})"
        );
        assert_eq!(format!("{:?}", Obj::Code(Arc::new(Code {
            argcount: 0,
            posonlyargcount: 1,
            kwonlyargcount: 2,
            nlocals: 3,
            stacksize: 4,
            flags: CodeFlags::NESTED | CodeFlags::COROUTINE,
            code: Arc::new(Vec::from(b"abc" as &[u8])),
            consts: Arc::new(vec![Obj::Bool(true)]),
            names: vec![],
            varnames: vec![Arc::new("a".to_owned())],
            freevars: vec![Arc::new("b".to_owned()), Arc::new("c".to_owned())],
            cellvars: vec![Arc::new("de".to_owned())],
            filename: Arc::new("xyz.py".to_owned()),
            name: Arc::new("fgh".to_owned()),
            firstlineno: 5,
            lnotab: Arc::new(vec![255, 0, 45, 127, 0, 73]),
        }))), "code(argcount=0, posonlyargcount=1, kwonlyargcount=2, nlocals=3, stacksize=4, flags=NESTED | COROUTINE, code=b\"abc\", consts=[True], names=[], varnames=[\"a\"], freevars=[\"b\", \"c\"], cellvars=[\"de\"], filename=\"xyz.py\", name=\"fgh\", firstlineno=5, lnotab=bytes([255, 0, 45, 127, 0, 73]))");
    }

    #[test]
    fn test_float_debug_repr() {
        assert_eq!(format!("{:?}", Obj::Float(1.23)), "1.23");
        assert_eq!(format!("{:?}", Obj::Float(f64::NAN)), "float('nan')");
        assert_eq!(format!("{:?}", Obj::Float(f64::INFINITY)), "float('inf')");
        assert_eq!(format!("{:?}", Obj::Float(-f64::INFINITY)), "-float('inf')");
        assert_eq!(format!("{:?}", Obj::Float(0.0)), "0.0");
        assert_eq!(format!("{:?}", Obj::Float(-0.0)), "-0.0");
    }

    #[test]
    fn test_complex_debug_repr() {
        assert_eq!(
            format!("{:?}", Obj::Complex(Complex { re: 2., im: 1. })),
            "(2+1j)"
        );
        assert_eq!(
            format!("{:?}", Obj::Complex(Complex { re: 0., im: 1. })),
            "1j"
        );
        assert_eq!(
            format!("{:?}", Obj::Complex(Complex { re: 2., im: 0. })),
            "(2+0j)"
        );
        assert_eq!(
            format!("{:?}", Obj::Complex(Complex { re: 0., im: 0. })),
            "0j"
        );
        assert_eq!(
            format!("{:?}", Obj::Complex(Complex { re: -2., im: 1. })),
            "(-2+1j)"
        );
        assert_eq!(
            format!("{:?}", Obj::Complex(Complex { re: -2., im: 0. })),
            "(-2+0j)"
        );
        assert_eq!(
            format!("{:?}", Obj::Complex(Complex { re: 2., im: -1. })),
            "(2-1j)"
        );
        assert_eq!(
            format!("{:?}", Obj::Complex(Complex { re: 0., im: -1. })),
            "-1j"
        );
        assert_eq!(
            format!("{:?}", Obj::Complex(Complex { re: -2., im: -1. })),
            "(-2-1j)"
        );
        assert_eq!(
            format!("{:?}", Obj::Complex(Complex { re: 0., im: -1. })),
            "-1j"
        );
        assert_eq!(
            format!("{:?}", Obj::Complex(Complex { re: -2., im: 0. })),
            "(-2+0j)"
        );
        assert_eq!(
            format!("{:?}", Obj::Complex(Complex { re: -0., im: 1. })),
            "(-0+1j)"
        );
        assert_eq!(
            format!("{:?}", Obj::Complex(Complex { re: -0., im: -1. })),
            "(-0-1j)"
        );
    }

    #[test]
    fn test_bytes_string_debug_repr() {
        assert_eq!(format!("{:?}", Obj::Bytes(Arc::new(Vec::from(
                            b"\x00\x01\x02\x03\x04\x05\x06\x07\x08\t\n\x0b\x0c\r\x0e\x0f\x10\x11\x12\x13\x14\x15\x16\x17\x18\x19\x1a\x1b\x1c\x1d\x1e\x1f !\"#$%&'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\\]^_`abcdefghijklmnopqrstuvwxyz{|}~\x7f\x80\x81\x82\x83\x84\x85\x86\x87\x88\x89\x8a\x8b\x8c\x8d\x8e\x8f\x90\x91\x92\x93\x94\x95\x96\x97\x98\x99\x9a\x9b\x9c\x9d\x9e\x9f\xa0\xa1\xa2\xa3\xa4\xa5\xa6\xa7\xa8\xa9\xaa\xab\xac\xad\xae\xaf\xb0\xb1\xb2\xb3\xb4\xb5\xb6\xb7\xb8\xb9\xba\xbb\xbc\xbd\xbe\xbf\xc0\xc1\xc2\xc3\xc4\xc5\xc6\xc7\xc8\xc9\xca\xcb\xcc\xcd\xce\xcf\xd0\xd1\xd2\xd3\xd4\xd5\xd6\xd7\xd8\xd9\xda\xdb\xdc\xdd\xde\xdf\xe0\xe1\xe2\xe3\xe4\xe5\xe6\xe7\xe8\xe9\xea\xeb\xec\xed\xee\xef\xf0\xf1\xf2\xf3\xf4\xf5\xf6\xf7\xf8\xf9\xfa\xfb\xfc\xfd\xfe" as &[u8]
                            )))),
        "b\"\\x00\\x01\\x02\\x03\\x04\\x05\\x06\\x07\\x08\\t\\n\\x0b\\x0c\\r\\x0e\\x0f\\x10\\x11\\x12\\x13\\x14\\x15\\x16\\x17\\x18\\x19\\x1a\\x1b\\x1c\\x1d\\x1e\\x1f !\\\"#$%&\\\'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\\\\]^_`abcdefghijklmnopqrstuvwxyz{|}~\\x7f\\x80\\x81\\x82\\x83\\x84\\x85\\x86\\x87\\x88\\x89\\x8a\\x8b\\x8c\\x8d\\x8e\\x8f\\x90\\x91\\x92\\x93\\x94\\x95\\x96\\x97\\x98\\x99\\x9a\\x9b\\x9c\\x9d\\x9e\\x9f\\xa0\\xa1\\xa2\\xa3\\xa4\\xa5\\xa6\\xa7\\xa8\\xa9\\xaa\\xab\\xac\\xad\\xae\\xaf\\xb0\\xb1\\xb2\\xb3\\xb4\\xb5\\xb6\\xb7\\xb8\\xb9\\xba\\xbb\\xbc\\xbd\\xbe\\xbf\\xc0\\xc1\\xc2\\xc3\\xc4\\xc5\\xc6\\xc7\\xc8\\xc9\\xca\\xcb\\xcc\\xcd\\xce\\xcf\\xd0\\xd1\\xd2\\xd3\\xd4\\xd5\\xd6\\xd7\\xd8\\xd9\\xda\\xdb\\xdc\\xdd\\xde\\xdf\\xe0\\xe1\\xe2\\xe3\\xe4\\xe5\\xe6\\xe7\\xe8\\xe9\\xea\\xeb\\xec\\xed\\xee\\xef\\xf0\\xf1\\xf2\\xf3\\xf4\\xf5\\xf6\\xf7\\xf8\\xf9\\xfa\\xfb\\xfc\\xfd\\xfe\""
        );
        assert_eq!(format!("{:?}", Obj::String(Arc::new(String::from(
                            "\x00\x01\x02\x03\x04\x05\x06\x07\x08\t\n\x0b\x0c\r\x0e\x0f\x10\x11\x12\x13\x14\x15\x16\x17\x18\x19\x1a\x1b\x1c\x1d\x1e\x1f !\"#$%&'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\\]^_`abcdefghijklmnopqrstuvwxyz{|}~\x7f")))),
                            "\"\\x00\\x01\\x02\\x03\\x04\\x05\\x06\\x07\\x08\\t\\n\\x0b\\x0c\\r\\x0e\\x0f\\x10\\x11\\x12\\x13\\x14\\x15\\x16\\x17\\x18\\x19\\x1a\\x1b\\x1c\\x1d\\x1e\\x1f !\\\"#$%&\\\'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\\\\]^_`abcdefghijklmnopqrstuvwxyz{|}~\\x7f\"");
    }
}

mod utils {
    use num_bigint::{BigUint, Sign};
    use num_traits::Zero;
    use std::cmp::Ordering;

    /// Based on `_PyLong_AsByteArray` in <https://github.com/python/cpython/blob/master/Objects/longobject.c>
    #[allow(clippy::cast_possible_truncation)]
    pub fn biguint_from_pylong_digits(digits: &[u16]) -> BigUint {
        if digits.is_empty() {
            return BigUint::zero();
        };
        assert!(digits[digits.len() - 1] != 0);
        let mut accum: u64 = 0;
        let mut accumbits: u8 = 0;
        let mut p = Vec::<u32>::new();
        for (i, &thisdigit) in digits.iter().enumerate() {
            accum |= u64::from(thisdigit) << accumbits;
            accumbits += if i == digits.len() - 1 {
                16 - (thisdigit.leading_zeros() as u8)
            } else {
                15
            };

            // Modified to get u32s instead of u8s.
            while accumbits >= 32 {
                p.push(accum as u32);
                accumbits -= 32;
                accum >>= 32;
            }
        }
        assert!(accumbits < 32);
        if accumbits > 0 {
            p.push(accum as u32);
        }
        BigUint::new(p)
    }

    pub fn sign_of<T: Ord + Zero>(x: &T) -> Sign {
        match x.cmp(&T::zero()) {
            Ordering::Less => Sign::Minus,
            Ordering::Equal => Sign::NoSign,
            Ordering::Greater => Sign::Plus,
        }
    }

    #[cfg(test)]
    mod test {
        use super::biguint_from_pylong_digits;
        use num_bigint::BigUint;

        #[allow(clippy::inconsistent_digit_grouping)]
        #[test]
        fn test_biguint_from_pylong_digits() {
            assert_eq!(
                biguint_from_pylong_digits(&[
                    0b000_1101_1100_0100,
                    0b110_1101_0010_0100,
                    0b001_0000_1001_1101
                ]),
                BigUint::from(0b001_0000_1001_1101_110_1101_0010_0100_000_1101_1100_0100_u64)
            );
        }
    }
}

#[allow(clippy::wildcard_imports)] // read::errors
pub mod read {
    pub mod errors {
        use error_chain::error_chain;

        error_chain! {
            foreign_links {
                Io(::std::io::Error);
                Utf8(::std::str::Utf8Error);
                FromUtf8(::std::string::FromUtf8Error);
                ParseFloat(::std::num::ParseFloatError);
            }

            errors {
                InvalidType(x: u8)
                RecursionLimitExceeded
                DigitOutOfRange(x: u16)
                UnnormalizedLong
                IsNull
                Unhashable(x: crate::Obj)
                TypeError(x: crate::Obj)
                InvalidRef
            }

            skip_msg_variant
        }
    }

    use self::errors::*;
    use crate::{utils, Code, CodeFlags, Depth, Obj, ObjHashable, Type};
    use num_bigint::BigInt;
    use num_complex::Complex;
    use num_traits::{FromPrimitive, Zero};
    use std::{
        collections::{HashMap, HashSet},
        convert::TryFrom,
        io::Read,
        str::FromStr,
        sync::{Arc, RwLock},
    };

    struct RFile<R: Read> {
        depth: Depth,
        readable: R,
        refs: Vec<Obj>,
        has_posonlyargcount: bool,
    }

    macro_rules! define_r {
        ($ident:ident -> $ty:ty; $n:literal) => {
            fn $ident(p: &mut RFile<impl Read>) -> Result<$ty> {
                let mut buf: [u8; $n] = [0; $n];
                p.readable.read_exact(&mut buf)?;
                Ok(<$ty>::from_le_bytes(buf))
            }
        };
    }

    define_r! { r_byte      -> u8 ; 1 }
    define_r! { r_short     -> u16; 2 }
    define_r! { r_long      -> u32; 4 }
    define_r! { r_long64    -> u64; 8 }
    define_r! { r_float_bin -> f64; 8 }

    fn r_bytes(n: usize, p: &mut RFile<impl Read>) -> Result<Vec<u8>> {
        let mut buf = Vec::new();
        buf.resize(n, 0);
        p.readable.read_exact(&mut buf)?;
        Ok(buf)
    }

    fn r_string(n: usize, p: &mut RFile<impl Read>) -> Result<String> {
        let buf = r_bytes(n, p)?;
        Ok(String::from_utf8(buf)?)
    }

    fn r_float_str(p: &mut RFile<impl Read>) -> Result<f64> {
        let n = r_byte(p)?;
        let s = r_string(n as usize, p)?;
        Ok(f64::from_str(&s)?)
    }

    // TODO: test
    /// May misbehave on 16-bit platforms.
    fn r_pylong(p: &mut RFile<impl Read>) -> Result<BigInt> {
        #[allow(clippy::cast_possible_wrap)]
        let n = r_long(p)? as i32;
        if n == 0 {
            return Ok(BigInt::zero());
        };
        #[allow(clippy::cast_sign_loss)]
        let size = n.wrapping_abs() as u32;
        let mut digits = Vec::<u16>::with_capacity(size as usize);
        for _ in 0..size {
            let d = r_short(p)?;
            if d > (1 << 15) {
                return Err(ErrorKind::DigitOutOfRange(d).into());
            }
            digits.push(d);
        }
        if digits[(size - 1) as usize] == 0 {
            return Err(ErrorKind::UnnormalizedLong.into());
        }
        Ok(BigInt::from_biguint(
            utils::sign_of(&n),
            utils::biguint_from_pylong_digits(&digits),
        ))
    }

    fn r_vec(n: usize, p: &mut RFile<impl Read>) -> Result<Vec<Obj>> {
        let mut vec = Vec::with_capacity(n);
        for _ in 0..n {
            vec.push(r_object_not_null(p)?);
        }
        Ok(vec)
    }

    fn r_hashmap(p: &mut RFile<impl Read>) -> Result<HashMap<ObjHashable, Obj>> {
        let mut map = HashMap::new();
        loop {
            match r_object(p)? {
                None => break,
                Some(key) => match r_object(p)? {
                    None => break,
                    Some(value) => {
                        map.insert(
                            ObjHashable::try_from(&key).map_err(ErrorKind::Unhashable)?,
                            value,
                        );
                    } // TODO
                },
            }
        }
        Ok(map)
    }

    fn r_hashset(n: usize, p: &mut RFile<impl Read>) -> Result<HashSet<ObjHashable>> {
        let mut set = HashSet::new();
        r_hashset_into(&mut set, n, p)?;
        Ok(set)
    }
    fn r_hashset_into(
        set: &mut HashSet<ObjHashable>,
        n: usize,
        p: &mut RFile<impl Read>,
    ) -> Result<()> {
        for _ in 0..n {
            set.insert(
                ObjHashable::try_from(&r_object_not_null(p)?).map_err(ErrorKind::Unhashable)?,
            );
        }
        Ok(())
    }

    #[allow(clippy::too_many_lines)]
    fn r_object(p: &mut RFile<impl Read>) -> Result<Option<Obj>> {
        let code: u8 = r_byte(p)?;
        let _depth_handle = p
            .depth
            .try_clone()
            .map_or(Err(ErrorKind::RecursionLimitExceeded), Ok)?;
        let (flag, type_) = {
            let flag: bool = (code & Type::FLAG_REF) != 0;
            let type_u8: u8 = code & !Type::FLAG_REF;
            let type_: Type =
                Type::from_u8(type_u8).map_or(Err(ErrorKind::InvalidType(type_u8)), Ok)?;
            (flag, type_)
        };
        let mut idx: Option<usize> = match type_ {
            // R_REF/r_ref_reserve before reading contents
            // See https://github.com/sollyucko/py-marshal/issues/2
            Type::SmallTuple | Type::Tuple | Type::List | Type::Dict | Type::Set | Type::FrozenSet | Type::Code if flag => {
                let i = p.refs.len();
                p.refs.push(Obj::None);
                Some(i)
            }
            _ => None,
        };
        #[allow(clippy::cast_possible_wrap)]
        let retval = match type_ {
            Type::Null => None,
            Type::None => Some(Obj::None),
            Type::StopIter => Some(Obj::StopIteration),
            Type::Ellipsis => Some(Obj::Ellipsis),
            Type::False => Some(Obj::Bool(false)),
            Type::True => Some(Obj::Bool(true)),
            Type::Int => Some(Obj::Long(Arc::new(BigInt::from(r_long(p)? as i32)))),
            Type::Int64 => Some(Obj::Long(Arc::new(BigInt::from(r_long64(p)? as i64)))),
            Type::Long => Some(Obj::Long(Arc::new(r_pylong(p)?))),
            Type::Float => Some(Obj::Float(r_float_str(p)?)),
            Type::BinaryFloat => Some(Obj::Float(r_float_bin(p)?)),
            Type::Complex => Some(Obj::Complex(Complex {
                re: r_float_str(p)?,
                im: r_float_str(p)?,
            })),
            Type::BinaryComplex => Some(Obj::Complex(Complex {
                re: r_float_bin(p)?,
                im: r_float_bin(p)?,
            })),
            Type::String => Some(Obj::Bytes(Arc::new(r_bytes(r_long(p)? as usize, p)?))),
            Type::AsciiInterned | Type::Ascii | Type::Interned | Type::Unicode => {
                Some(Obj::String(Arc::new(r_string(r_long(p)? as usize, p)?)))
            }
            Type::ShortAsciiInterned | Type::ShortAscii => {
                Some(Obj::String(Arc::new(r_string(r_byte(p)? as usize, p)?)))
            }
            Type::SmallTuple => Some(Obj::Tuple(Arc::new(r_vec(r_byte(p)? as usize, p)?))),
            Type::Tuple => Some(Obj::Tuple(Arc::new(r_vec(r_long(p)? as usize, p)?))),
            Type::List => Some(Obj::List(Arc::new(RwLock::new(r_vec(
                r_long(p)? as usize,
                p,
            )?)))),
            Type::Set => {
                let set = Arc::new(RwLock::new(HashSet::new()));

                if flag {
                    idx = Some(p.refs.len());
                    p.refs.push(Obj::Set(Arc::clone(&set)));
                }

                r_hashset_into(&mut *set.write().unwrap(), r_long(p)? as usize, p)?;
                Some(Obj::Set(set))
            }
            Type::FrozenSet => Some(Obj::FrozenSet(Arc::new(r_hashset(r_long(p)? as usize, p)?))),
            Type::Dict => Some(Obj::Dict(Arc::new(RwLock::new(r_hashmap(p)?)))),
            Type::Code => Some(Obj::Code(Arc::new(Code {
                argcount: r_long(p)?,
                posonlyargcount: if p.has_posonlyargcount { r_long(p)? } else { 0 },
                kwonlyargcount: r_long(p)?,
                nlocals: r_long(p)?,
                stacksize: r_long(p)?,
                flags: CodeFlags::from_bits_truncate(r_long(p)?),
                code: r_object_extract_bytes(p)?,
                consts: r_object_extract_tuple(p)?,
                names: r_object_extract_tuple_string(p)?,
                varnames: r_object_extract_tuple_string(p)?,
                freevars: r_object_extract_tuple_string(p)?,
                cellvars: r_object_extract_tuple_string(p)?,
                filename: r_object_extract_string(p)?,
                name: r_object_extract_string(p)?,
                firstlineno: r_long(p)?,
                lnotab: r_object_extract_bytes(p)?,
            }))),

            Type::Ref => {
                let n = r_long(p)? as usize;
                let result = p.refs.get(n).ok_or(ErrorKind::InvalidRef)?.clone();
                if result.is_none() {
                    return Err(ErrorKind::InvalidRef.into());
                } else {
                    Some(result)
                }
            }
            Type::Unknown => return Err(ErrorKind::InvalidType(Type::Unknown as u8).into()),
        };
        match (&retval, idx) {
            (None, _)
            | (Some(Obj::None), _)
            | (Some(Obj::StopIteration), _)
            | (Some(Obj::Ellipsis), _)
            | (Some(Obj::Bool(_)), _) => {}
            (Some(x), Some(i)) if flag => {
                p.refs[i] = x.clone();
            }
            (Some(x), None) if flag => {
                p.refs.push(x.clone());
            }
            (Some(_), _) => {}
        };
        Ok(retval)
    }

    fn r_object_not_null(p: &mut RFile<impl Read>) -> Result<Obj> {
        Ok(r_object(p)?.ok_or(ErrorKind::IsNull)?)
    }
    fn r_object_extract_string(p: &mut RFile<impl Read>) -> Result<Arc<String>> {
        Ok(r_object_not_null(p)?
            .extract_string()
            .map_err(ErrorKind::TypeError)?)
    }
    fn r_object_extract_bytes(p: &mut RFile<impl Read>) -> Result<Arc<Vec<u8>>> {
        Ok(r_object_not_null(p)?
            .extract_bytes()
            .map_err(ErrorKind::TypeError)?)
    }
    fn r_object_extract_tuple(p: &mut RFile<impl Read>) -> Result<Arc<Vec<Obj>>> {
        Ok(r_object_not_null(p)?
            .extract_tuple()
            .map_err(ErrorKind::TypeError)?)
    }
    fn r_object_extract_tuple_string(p: &mut RFile<impl Read>) -> Result<Vec<Arc<String>>> {
        Ok(r_object_extract_tuple(p)?
            .iter()
            .map(|x| {
                x.clone()
                    .extract_string()
                    .map_err(|o: Obj| Error::from(ErrorKind::TypeError(o)))
            })
            .collect::<Result<Vec<Arc<String>>>>()?)
    }

    fn read_object(p: &mut RFile<impl Read>) -> Result<Obj> {
        r_object_not_null(p)
    }

    #[derive(Copy, Clone, Debug)]
    pub struct MarshalLoadExOptions {
        pub has_posonlyargcount: bool,
    }
    /// Assume latest version
    impl Default for MarshalLoadExOptions {
        fn default() -> Self {
            Self {
                has_posonlyargcount: true,
            }
        }
    }

    /// # Errors
    /// See [`ErrorKind`].
    pub fn marshal_load_ex(readable: impl Read, opts: MarshalLoadExOptions) -> Result<Obj> {
        let mut rf = RFile {
            depth: Depth::new(),
            readable,
            refs: Vec::<Obj>::new(),
            has_posonlyargcount: opts.has_posonlyargcount,
        };
        read_object(&mut rf)
    }

    /// # Errors
    /// See [`ErrorKind`].
    pub fn marshal_load(readable: impl Read) -> Result<Obj> {
        marshal_load_ex(readable, MarshalLoadExOptions::default())
    }

    /// Allows coercion from array reference to slice.
    /// # Errors
    /// See [`ErrorKind`].
    pub fn marshal_loads(bytes: &[u8]) -> Result<Obj> {
        marshal_load(bytes)
    }

    // Ported from <https://github.com/python/cpython/blob/master/Lib/test/test_marshal.py>
    #[cfg(test)]
    mod test {
        use super::{
            errors, marshal_load, marshal_load_ex, marshal_loads, Code, CodeFlags,
            MarshalLoadExOptions, Obj, ObjHashable,
        };
        use num_bigint::BigInt;
        use num_traits::Pow;
        use std::{
            io::{self, Read},
            ops::Deref,
            sync::Arc,
        };

        macro_rules! assert_match {
            ($expr:expr, $pat:pat) => {
                match $expr {
                    $pat => {}
                    _ => panic!(),
                }
            };
        }

        fn load_unwrap(r: impl Read) -> Obj {
            marshal_load(r).unwrap()
        }

        fn loads_unwrap(s: &[u8]) -> Obj {
            load_unwrap(s)
        }

        #[test]
        fn test_ints() {
            assert_eq!(BigInt::parse_bytes(b"85070591730234615847396907784232501249", 10).unwrap(), *loads_unwrap(b"l\t\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\xf0\x7f\xff\x7f\xff\x7f\xff\x7f?\x00").extract_long().unwrap());
        }

        #[allow(clippy::unreadable_literal)]
        #[test]
        fn test_int64() {
            for mut base in [i64::MAX, i64::MIN, -i64::MAX, -(i64::MIN >> 1)]
                .iter()
                .copied()
            {
                while base != 0 {
                    let mut s = Vec::<u8>::new();
                    s.push(b'I');
                    s.extend_from_slice(&base.to_le_bytes());
                    assert_eq!(
                        BigInt::from(base),
                        *loads_unwrap(&s).extract_long().unwrap()
                    );

                    if base == -1 {
                        base = 0
                    } else {
                        base >>= 1
                    }
                }
            }

            assert_eq!(
                BigInt::from(0x1032547698badcfe_i64),
                *loads_unwrap(b"I\xfe\xdc\xba\x98\x76\x54\x32\x10")
                    .extract_long()
                    .unwrap()
            );
            assert_eq!(
                BigInt::from(-0x1032547698badcff_i64),
                *loads_unwrap(b"I\x01\x23\x45\x67\x89\xab\xcd\xef")
                    .extract_long()
                    .unwrap()
            );
            assert_eq!(
                BigInt::from(0x7f6e5d4c3b2a1908_i64),
                *loads_unwrap(b"I\x08\x19\x2a\x3b\x4c\x5d\x6e\x7f")
                    .extract_long()
                    .unwrap()
            );
            assert_eq!(
                BigInt::from(-0x7f6e5d4c3b2a1909_i64),
                *loads_unwrap(b"I\xf7\xe6\xd5\xc4\xb3\xa2\x91\x80")
                    .extract_long()
                    .unwrap()
            );
        }

        #[test]
        fn test_bool() {
            assert_eq!(true, loads_unwrap(b"T").extract_bool().unwrap());
            assert_eq!(false, loads_unwrap(b"F").extract_bool().unwrap());
        }

        #[allow(clippy::float_cmp, clippy::cast_precision_loss)]
        #[test]
        fn test_floats() {
            assert_eq!(
                (i64::MAX as f64) * 3.7e250,
                loads_unwrap(b"g\x11\x9f6\x98\xd2\xab\xe4w")
                    .extract_float()
                    .unwrap()
            );
        }

        #[test]
        fn test_unicode() {
            assert_eq!("", *loads_unwrap(b"\xda\x00").extract_string().unwrap());
            assert_eq!(
                "Andr\u{e8} Previn",
                *loads_unwrap(b"u\r\x00\x00\x00Andr\xc3\xa8 Previn")
                    .extract_string()
                    .unwrap()
            );
            assert_eq!(
                "abc",
                *loads_unwrap(b"\xda\x03abc").extract_string().unwrap()
            );
            assert_eq!(
                " ".repeat(10_000),
                *loads_unwrap(&[b"a\x10'\x00\x00" as &[u8], &[b' '; 10_000]].concat())
                    .extract_string()
                    .unwrap()
            );
        }

        #[test]
        fn test_string() {
            assert_eq!("", *loads_unwrap(b"\xda\x00").extract_string().unwrap());
            assert_eq!(
                "Andr\u{e8} Previn",
                *loads_unwrap(b"\xf5\r\x00\x00\x00Andr\xc3\xa8 Previn")
                    .extract_string()
                    .unwrap()
            );
            assert_eq!(
                "abc",
                *loads_unwrap(b"\xda\x03abc").extract_string().unwrap()
            );
            assert_eq!(
                " ".repeat(10_000),
                *loads_unwrap(&[b"\xe1\x10'\x00\x00" as &[u8], &[b' '; 10_000]].concat())
                    .extract_string()
                    .unwrap()
            );
        }

        #[test]
        fn test_bytes() {
            assert_eq!(
                b"",
                &loads_unwrap(b"\xf3\x00\x00\x00\x00")
                    .extract_bytes()
                    .unwrap()[..]
            );
            assert_eq!(
                b"Andr\xe8 Previn",
                &loads_unwrap(b"\xf3\x0c\x00\x00\x00Andr\xe8 Previn")
                    .extract_bytes()
                    .unwrap()[..]
            );
            assert_eq!(
                b"abc",
                &loads_unwrap(b"\xf3\x03\x00\x00\x00abc")
                    .extract_bytes()
                    .unwrap()[..]
            );
            assert_eq!(
                b" ".repeat(10_000),
                &loads_unwrap(&[b"\xf3\x10'\x00\x00" as &[u8], &[b' '; 10_000]].concat())
                    .extract_bytes()
                    .unwrap()[..]
            );
        }

        #[test]
        fn test_exceptions() {
            loads_unwrap(b"S").extract_stop_iteration().unwrap();
        }

        fn assert_test_exceptions_code_valid(code: &Code) {
            assert_eq!(code.argcount, 1);
            assert!(code.cellvars.is_empty());
            assert_eq!(*code.code, b"t\x00\xa0\x01t\x00\xa0\x02t\x03\xa1\x01\xa1\x01}\x01|\x00\xa0\x04t\x03|\x01\xa1\x02\x01\x00d\x00S\x00");
            assert_eq!(code.consts.len(), 1);
            assert!(code.consts[0].is_none());
            assert_eq!(*code.filename, "<string>");
            assert_eq!(code.firstlineno, 3);
            assert_eq!(
                code.flags,
                CodeFlags::NOFREE | CodeFlags::NEWLOCALS | CodeFlags::OPTIMIZED
            );
            assert!(code.freevars.is_empty());
            assert_eq!(code.kwonlyargcount, 0);
            assert_eq!(*code.lnotab, b"\x00\x01\x10\x01");
            assert_eq!(*code.name, "test_exceptions");
            assert!(code.names.iter().map(Deref::deref).eq(vec![
                "marshal",
                "loads",
                "dumps",
                "StopIteration",
                "assertEqual"
            ]
            .iter()));
            assert_eq!(code.nlocals, 2);
            assert_eq!(code.stacksize, 5);
            assert!(code
                .varnames
                .iter()
                .map(Deref::deref)
                .eq(vec!["self", "new"].iter()));
        }

        #[test]
        fn test_code() {
            // ExceptionTestCase.test_exceptions
            // { 'co_argcount': 1, 'co_cellvars': (), 'co_code': b't\x00\xa0\x01t\x00\xa0\x02t\x03\xa1\x01\xa1\x01}\x01|\x00\xa0\x04t\x03|\x01\xa1\x02\x01\x00d\x00S\x00', 'co_consts': (None,), 'co_filename': '<string>', 'co_firstlineno': 3, 'co_flags': 67, 'co_freevars': (), 'co_kwonlyargcount': 0, 'co_lnotab': b'\x00\x01\x10\x01', 'co_name': 'test_exceptions', 'co_names': ('marshal', 'loads', 'dumps', 'StopIteration', 'assertEqual'), 'co_nlocals': 2, 'co_stacksize': 5, 'co_varnames': ('self', 'new') }
            let mut input: &[u8] = b"\xe3\x01\x00\x00\x00\x00\x00\x00\x00\x02\x00\x00\x00\x05\x00\x00\x00C\x00\x00\x00s \x00\x00\x00t\x00\xa0\x01t\x00\xa0\x02t\x03\xa1\x01\xa1\x01}\x01|\x00\xa0\x04t\x03|\x01\xa1\x02\x01\x00d\x00S\x00)\x01N)\x05\xda\x07marshal\xda\x05loads\xda\x05dumps\xda\rStopIteration\xda\x0bassertEqual)\x02\xda\x04self\xda\x03new\xa9\x00r\x08\x00\x00\x00\xda\x08<string>\xda\x0ftest_exceptions\x03\x00\x00\x00s\x04\x00\x00\x00\x00\x01\x10\x01";
            println!("{}", input.len());
            let code_result = marshal_load_ex(
                &mut input,
                MarshalLoadExOptions {
                    has_posonlyargcount: false,
                },
            );
            println!("{}", input.len());
            let code = code_result.unwrap().extract_code().unwrap();
            assert_test_exceptions_code_valid(&code);
        }

        #[test]
        fn test_many_codeobjects() {
            let mut input: &[u8] = &[b"(\x88\x13\x00\x00\xe3\x01\x00\x00\x00\x00\x00\x00\x00\x02\x00\x00\x00\x05\x00\x00\x00C\x00\x00\x00s \x00\x00\x00t\x00\xa0\x01t\x00\xa0\x02t\x03\xa1\x01\xa1\x01}\x01|\x00\xa0\x04t\x03|\x01\xa1\x02\x01\x00d\x00S\x00)\x01N)\x05\xda\x07marshal\xda\x05loads\xda\x05dumps\xda\rStopIteration\xda\x0bassertEqual)\x02\xda\x04self\xda\x03new\xa9\x00r\x08\x00\x00\x00\xda\x08<string>\xda\x0ftest_exceptions\x03\x00\x00\x00s\x04\x00\x00\x00\x00\x01\x10\x01" as &[u8], &b"r\x00\x00\x00\x00".repeat(4999)].concat();
            let result = marshal_load_ex(
                &mut input,
                MarshalLoadExOptions {
                    has_posonlyargcount: false,
                },
            );
            let tuple = result.unwrap().extract_tuple().unwrap();
            for o in &*tuple {
                assert_test_exceptions_code_valid(&o.clone().extract_code().unwrap());
            }
        }

        #[test]
        fn test_different_filenames() {
            let mut input: &[u8] = b")\x02c\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00@\x00\x00\x00s\x08\x00\x00\x00e\x00\x01\x00d\x00S\x00)\x01N)\x01\xda\x01x\xa9\x00r\x01\x00\x00\x00r\x01\x00\x00\x00\xda\x02f1\xda\x08<module>\x01\x00\x00\x00\xf3\x00\x00\x00\x00c\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00@\x00\x00\x00s\x08\x00\x00\x00e\x00\x01\x00d\x00S\x00)\x01N)\x01\xda\x01yr\x01\x00\x00\x00r\x01\x00\x00\x00r\x01\x00\x00\x00\xda\x02f2r\x03\x00\x00\x00\x01\x00\x00\x00r\x04\x00\x00\x00";
            println!("{}", input.len());
            let result = marshal_load_ex(
                &mut input,
                MarshalLoadExOptions {
                    has_posonlyargcount: false,
                },
            );
            println!("{}", input.len());
            let tuple = result.unwrap().extract_tuple().unwrap();
            assert_eq!(tuple.len(), 2);
            assert_eq!(*tuple[0].clone().extract_code().unwrap().filename, "f1");
            assert_eq!(*tuple[1].clone().extract_code().unwrap().filename, "f2");
        }

        #[allow(clippy::float_cmp)]
        #[test]
        fn test_dict() {
            let mut input: &[u8] = b"{\xda\x07astring\xfa\x10foo@bar.baz.spam\xda\x06afloat\xe7H\xe1z\x14ns\xbc@\xda\x05anint\xe9\x00\x00\x10\x00\xda\nashortlong\xe9\x02\x00\x00\x00\xda\x05alist[\x01\x00\x00\x00\xfa\x07.zyx.41\xda\x06atuple\xa9\n\xfa\x07.zyx.41r\x0c\x00\x00\x00r\x0c\x00\x00\x00r\x0c\x00\x00\x00r\x0c\x00\x00\x00r\x0c\x00\x00\x00r\x0c\x00\x00\x00r\x0c\x00\x00\x00r\x0c\x00\x00\x00r\x0c\x00\x00\x00\xda\x08abooleanF\xda\x08aunicode\xf5\r\x00\x00\x00Andr\xc3\xa8 Previn0";
            println!("{}", input.len());
            let result = marshal_load(&mut input);
            println!("{}", input.len());
            let dict_ref = result.unwrap().extract_dict().unwrap();
            let dict = dict_ref.try_read().unwrap();
            assert_eq!(dict.len(), 8);
            assert_eq!(
                *dict[&ObjHashable::String(Arc::new("astring".to_owned()))]
                    .clone()
                    .extract_string()
                    .unwrap(),
                "foo@bar.baz.spam"
            );
            assert_eq!(
                dict[&ObjHashable::String(Arc::new("afloat".to_owned()))]
                    .clone()
                    .extract_float()
                    .unwrap(),
                7283.43_f64
            );
            assert_eq!(
                *dict[&ObjHashable::String(Arc::new("anint".to_owned()))]
                    .clone()
                    .extract_long()
                    .unwrap(),
                BigInt::from(2).pow(20_u8)
            );
            assert_eq!(
                *dict[&ObjHashable::String(Arc::new("ashortlong".to_owned()))]
                    .clone()
                    .extract_long()
                    .unwrap(),
                BigInt::from(2)
            );

            let list_ref = dict[&ObjHashable::String(Arc::new("alist".to_owned()))]
                .clone()
                .extract_list()
                .unwrap();
            let list = list_ref.try_read().unwrap();
            assert_eq!(list.len(), 1);
            assert_eq!(*list[0].clone().extract_string().unwrap(), ".zyx.41");

            let tuple = dict[&ObjHashable::String(Arc::new("atuple".to_owned()))]
                .clone()
                .extract_tuple()
                .unwrap();
            assert_eq!(tuple.len(), 10);
            for o in &*tuple {
                assert_eq!(*o.clone().extract_string().unwrap(), ".zyx.41");
            }
            assert_eq!(
                dict[&ObjHashable::String(Arc::new("aboolean".to_owned()))]
                    .clone()
                    .extract_bool()
                    .unwrap(),
                false
            );
            assert_eq!(
                *dict[&ObjHashable::String(Arc::new("aunicode".to_owned()))]
                    .clone()
                    .extract_string()
                    .unwrap(),
                "Andr\u{e8} Previn"
            );
        }

        /// Tests hash implementation
        #[test]
        fn test_dict_tuple_key() {
            let dict = loads_unwrap(b"{\xa9\x02\xda\x01a\xda\x01b\xda\x01c0")
                .extract_dict()
                .unwrap();
            assert_eq!(dict.read().unwrap().len(), 1);
            assert_eq!(
                *dict.read().unwrap()[&ObjHashable::Tuple(Arc::new(vec![
                    ObjHashable::String(Arc::new("a".to_owned())),
                    ObjHashable::String(Arc::new("b".to_owned()))
                ]))]
                    .clone()
                    .extract_string()
                    .unwrap(),
                "c"
            );
        }

        // TODO: test_list and test_tuple

        #[test]
        fn test_sets() {
            let set = loads_unwrap(b"<\x08\x00\x00\x00\xda\x05alist\xda\x08aboolean\xda\x07astring\xda\x08aunicode\xda\x06afloat\xda\x05anint\xda\x06atuple\xda\nashortlong").extract_set().unwrap();
            assert_eq!(set.read().unwrap().len(), 8);
            let frozenset = loads_unwrap(b">\x08\x00\x00\x00\xda\x06atuple\xda\x08aunicode\xda\x05anint\xda\x08aboolean\xda\x06afloat\xda\x05alist\xda\nashortlong\xda\x07astring").extract_frozenset().unwrap();
            assert_eq!(frozenset.len(), 8);
            // TODO: check values
        }

        // TODO: test_bytearray, test_memoryview, test_array

        #[test]
        fn test_patch_873224() {
            assert_match!(
                marshal_loads(b"0").unwrap_err().kind(),
                errors::ErrorKind::IsNull
            );
            let f_err = marshal_loads(b"f").unwrap_err();
            match f_err.kind() {
                errors::ErrorKind::Io(io_err) => {
                    assert_eq!(io_err.kind(), io::ErrorKind::UnexpectedEof);
                }
                _ => panic!(),
            }
            let int_err =
                marshal_loads(b"l\x05\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00 ").unwrap_err();
            match int_err.kind() {
                errors::ErrorKind::Io(io_err) => {
                    assert_eq!(io_err.kind(), io::ErrorKind::UnexpectedEof);
                }
                _ => panic!(),
            }
        }

        #[test]
        fn test_fuzz() {
            for i in 0..=u8::MAX {
                println!("{:?}", marshal_loads(&[i]));
            }
        }

        /// Warning: this has to be run on a release build to avoid a stack overflow.
        #[cfg(not(debug_assertions))]
        #[test]
        fn test_loads_recursion() {
            loads_unwrap(&[&b")\x01".repeat(100)[..], b"N"].concat());
            loads_unwrap(&[&b"(\x01\x00\x00\x00".repeat(100)[..], b"N"].concat());
            loads_unwrap(&[&b"[\x01\x00\x00\x00".repeat(100)[..], b"N"].concat());
            loads_unwrap(&[&b"{N".repeat(100)[..], b"N", &b"0".repeat(100)[..]].concat());
            loads_unwrap(&[&b">\x01\x00\x00\x00".repeat(100)[..], b"N"].concat());

            assert_match!(
                marshal_loads(&[&b")\x01".repeat(1048576)[..], b"N"].concat())
                    .unwrap_err()
                    .kind(),
                errors::ErrorKind::RecursionLimitExceeded
            );
            assert_match!(
                marshal_loads(&[&b"(\x01\x00\x00\x00".repeat(1048576)[..], b"N"].concat())
                    .unwrap_err()
                    .kind(),
                errors::ErrorKind::RecursionLimitExceeded
            );
            assert_match!(
                marshal_loads(&[&b"[\x01\x00\x00\x00".repeat(1048576)[..], b"N"].concat())
                    .unwrap_err()
                    .kind(),
                errors::ErrorKind::RecursionLimitExceeded
            );
            assert_match!(
                marshal_loads(
                    &[&b"{N".repeat(1048576)[..], b"N", &b"0".repeat(1048576)[..]].concat()
                )
                .unwrap_err()
                .kind(),
                errors::ErrorKind::RecursionLimitExceeded
            );
            assert_match!(
                marshal_loads(&[&b">\x01\x00\x00\x00".repeat(1048576)[..], b"N"].concat())
                    .unwrap_err()
                    .kind(),
                errors::ErrorKind::RecursionLimitExceeded
            );
        }

        #[test]
        fn test_invalid_longs() {
            assert_match!(
                marshal_loads(b"l\x02\x00\x00\x00\x00\x00\x00\x00")
                    .unwrap_err()
                    .kind(),
                errors::ErrorKind::UnnormalizedLong
            );
        }
        
        // See https://github.com/sollyucko/py-marshal/issues/2
        #[test]
        fn test_issue_2_ref_demarshalling_ordering_previously_broken() {
            let list_ref = marshal_loads(b"\xdb\x02\x00\x00\x00\xda\x01ar\x01\x00\x00\x00").unwrap().extract_list().unwrap();
            let list = list_ref.try_read().unwrap();
            assert_eq!(list.len(), 2);
            assert_eq!(*list[0].clone().extract_string().unwrap(), "a");
            assert_eq!(*list[1].clone().extract_string().unwrap(), "a");
        }
        #[test]
        fn test_issue_2_ref_demarshalling_ordering_previously_working() {
            let list_ref = marshal_loads(b"[\x02\x00\x00\x00\xda\x01ar\x00\x00\x00\x00").unwrap().extract_list().unwrap();
            let list = list_ref.try_read().unwrap();
            assert_eq!(list.len(), 2);
            assert_eq!(*list[0].clone().extract_string().unwrap(), "a");
            assert_eq!(*list[1].clone().extract_string().unwrap(), "a");
        }
    }
}
