/// Ported from <https://github.com/python/cpython/blob/master/Python/marshal.c>
use bitflags::bitflags;
use num_bigint::BigInt;
use num_complex::Complex;
use num_derive::{FromPrimitive, ToPrimitive};
use std::{
    collections::{HashMap, HashSet},
    convert::TryFrom,
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
    const MAX: usize = 2000;

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
    argcount:        u32,
    posonlyargcount: u32,
    kwonlyargcount:  u32,
    nlocals:         u32,
    stacksize:       u32,
    flags:           CodeFlags,
    code:            Arc<Vec<u8>>,
    consts:          Arc<Vec<Obj>>,
    names:           Arc<Vec<String>>,
    varnames:        Arc<Vec<String>>,
    freevars:        Arc<Vec<String>>,
    cellvars:        Arc<Vec<String>>,
    filename:        Arc<String>,
    name:            Arc<String>,
    firstlineno:     u32,
    lnotab:          Arc<Vec<u8>>,
}

#[rustfmt::skip]
#[derive(Clone, Debug)]
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
        define_extract! { $extract_fn -> Arc<$ret> { $variant(x) => Arc::clone(x) } }
    };
    ($extract_fn:ident($variant:ident) -> ArcRwLock<$ret:ty>) => {
        define_extract! { $extract_fn -> ArcRwLock<$ret> { $variant(x) => Arc::clone(x) } }
    };
    ($extract_fn:ident($variant:ident) -> $ret:ty) => {
        define_extract! { $extract_fn -> $ret { $variant(x) => *x } }
    };
    ($extract_fn:ident -> $ret:ty { $variant:ident$(($($pat:pat),+))? => $expr:expr }) => {
        /// # Errors
        /// Returns a reference to self if extraction fails
        pub fn $extract_fn(&self) -> Result<$ret, &Self> {
            if let Self::$variant$(($($pat),+))? = self {
                Ok($expr)
            } else {
                Err(self)
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

#[derive(PartialEq, Eq, Hash, Clone, Debug)]
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
            // immutable collections
            Type::SmallTuple | Type::Tuple | Type::FrozenSet | Type::Code if flag => {
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
                println!("{} {} {:?}", n, p.refs.len(), p.refs);
                let result = p.refs.get(n).ok_or(ErrorKind::InvalidRef)?.clone();
                if result.extract_none().is_ok() {
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
            .map_err(Obj::clone)
            .map_err(ErrorKind::TypeError)?)
    }
    fn r_object_extract_bytes(p: &mut RFile<impl Read>) -> Result<Arc<Vec<u8>>> {
        Ok(r_object_not_null(p)?
            .extract_bytes()
            .map_err(Obj::clone)
            .map_err(ErrorKind::TypeError)?)
    }
    fn r_object_extract_tuple(p: &mut RFile<impl Read>) -> Result<Arc<Vec<Obj>>> {
        Ok(r_object_not_null(p)?
            .extract_tuple()
            .map_err(Obj::clone)
            .map_err(ErrorKind::TypeError)?)
    }
    fn r_object_extract_tuple_string(p: &mut RFile<impl Read>) -> Result<Arc<Vec<String>>> {
        Ok(Arc::new(
            r_object_extract_tuple(p)?
                .iter()
                .map(|x| {
                    x.extract_string()
                        .as_ref()
                        .map(|s: &Arc<String>| (**s).clone())
                        .map_err(|o: &&Obj| Error::from(ErrorKind::TypeError(Obj::clone(*o))))
                })
                .collect::<Result<Vec<String>>>()?,
        ))
    }

    fn read_object(p: &mut RFile<impl Read>) -> Result<Option<Obj>> {
        r_object(p)
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
    pub fn marshal_load_ex(readable: impl Read, opts: MarshalLoadExOptions) -> Result<Option<Obj>> {
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
    pub fn marshal_load(readable: impl Read) -> Result<Option<Obj>> {
        marshal_load_ex(readable, MarshalLoadExOptions::default())
    }

    /// Allows coercion from array reference to slice.
    /// # Errors
    /// See [`ErrorKind`].
    pub fn marshal_loads(bytes: &[u8]) -> Result<Option<Obj>> {
        marshal_load(bytes)
    }

    /// Ported from <https://github.com/python/cpython/blob/master/Lib/test/test_marshal.py>
    #[cfg(test)]
    mod test {
        use super::{
            marshal_load, marshal_load_ex, Code, CodeFlags, MarshalLoadExOptions, Obj, ObjHashable,
        };
        use num_bigint::BigInt;
        use num_traits::Pow;
        use std::{io::Read, sync::Arc};

        fn load_unwrap(r: impl Read) -> Obj {
            marshal_load(r).unwrap().unwrap()
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
            code.consts[0].extract_none().unwrap();
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
            assert_eq!(
                *code.names,
                vec!["marshal", "loads", "dumps", "StopIteration", "assertEqual"]
            );
            assert_eq!(code.nlocals, 2);
            assert_eq!(code.stacksize, 5);
            assert_eq!(*code.varnames, vec!["self", "new"]);
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
            let code = code_result.unwrap().unwrap().extract_code().unwrap();
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
            let tuple = result.unwrap().unwrap().extract_tuple().unwrap();
            for o in &*tuple {
                assert_test_exceptions_code_valid(&o.extract_code().unwrap());
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
            let tuple = result.unwrap().unwrap().extract_tuple().unwrap();
            assert_eq!(tuple.len(), 2);
            assert_eq!(*tuple[0].extract_code().unwrap().filename, "f1");
            assert_eq!(*tuple[1].extract_code().unwrap().filename, "f2");
        }

        #[allow(clippy::float_cmp)]
        #[test]
        fn test_dict() {
            let mut input: &[u8] = b"{\xda\x07astring\xfa\x10foo@bar.baz.spam\xda\x06afloat\xe7H\xe1z\x14ns\xbc@\xda\x05anint\xe9\x00\x00\x10\x00\xda\nashortlong\xe9\x02\x00\x00\x00\xda\x05alist[\x01\x00\x00\x00\xfa\x07.zyx.41\xda\x06atuple\xa9\n\xfa\x07.zyx.41r\x0c\x00\x00\x00r\x0c\x00\x00\x00r\x0c\x00\x00\x00r\x0c\x00\x00\x00r\x0c\x00\x00\x00r\x0c\x00\x00\x00r\x0c\x00\x00\x00r\x0c\x00\x00\x00r\x0c\x00\x00\x00\xda\x08abooleanF\xda\x08aunicode\xf5\r\x00\x00\x00Andr\xc3\xa8 Previn0";
            println!("{}", input.len());
            let result = marshal_load(&mut input);
            println!("{}", input.len());
            let dict_ref = result.unwrap().unwrap().extract_dict().unwrap();
            let dict = dict_ref.try_read().unwrap();
            assert_eq!(dict.len(), 8);
            assert_eq!(
                *dict[&ObjHashable::String(Arc::new("astring".to_owned()))]
                    .extract_string()
                    .unwrap(),
                "foo@bar.baz.spam"
            );
            assert_eq!(
                dict[&ObjHashable::String(Arc::new("afloat".to_owned()))]
                    .extract_float()
                    .unwrap(),
                7283.43_f64
            );
            assert_eq!(
                *dict[&ObjHashable::String(Arc::new("anint".to_owned()))]
                    .extract_long()
                    .unwrap(),
                BigInt::from(2).pow(20_u8)
            );
            assert_eq!(
                *dict[&ObjHashable::String(Arc::new("ashortlong".to_owned()))]
                    .extract_long()
                    .unwrap(),
                BigInt::from(2)
            );

            let list_ref = dict[&ObjHashable::String(Arc::new("alist".to_owned()))]
                .extract_list()
                .unwrap();
            let list = list_ref.try_read().unwrap();
            assert_eq!(list.len(), 1);
            assert_eq!(*list[0].extract_string().unwrap(), ".zyx.41");

            let tuple = dict[&ObjHashable::String(Arc::new("atuple".to_owned()))]
                .extract_tuple()
                .unwrap();
            assert_eq!(tuple.len(), 10);
            for o in &*tuple {
                assert_eq!(*o.extract_string().unwrap(), ".zyx.41");
            }
            assert_eq!(
                dict[&ObjHashable::String(Arc::new("aboolean".to_owned()))]
                    .extract_bool()
                    .unwrap(),
                false
            );
            assert_eq!(
                *dict[&ObjHashable::String(Arc::new("aunicode".to_owned()))]
                    .extract_string()
                    .unwrap(),
                "Andr\u{e8} Previn"
            );
        }
    }
}
