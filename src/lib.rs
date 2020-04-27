/// Ported from https://github.com/python/cpython/blob/master/Python/marshal.c
use num_bigint::BigInt;
use num_complex::Complex;
use num_derive::{FromPrimitive, ToPrimitive};
use num_traits::FromPrimitive;
use std::collections::{HashMap, HashSet};
use std::convert::TryFrom;
use std::hash::{Hash, Hasher};
use std::iter::FromIterator;
use std::rc::Rc;

#[derive(FromPrimitive, ToPrimitive)]
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

struct Depth(Rc<()>);
impl Depth {
    const MAX: usize = 2000;

    #[must_use]
    pub fn new() -> Self {
        Self(Rc::new(()))
    }

    pub fn try_clone(&self) -> Option<Self> {
        if Rc::strong_count(&self.0) > Self::MAX {
            None
        } else {
            Some(Self(self.0.clone()))
        }
    }
}

//#[derive(PartialEq, Eq)]
#[derive(Clone)]
pub enum Obj {
    None,
    StopIteration,
    Ellipsis,
    Bool(bool),
    Long(Rc<BigInt>),
    Float(f64 /*HashF64*/),
    Complex(Complex<f64 /*HashF64*/>),
    String(Rc<String>),
    Tuple(Rc<Vec<Obj>>),
    List(Rc<Vec<Obj>>),
    Dict(Rc<HashMap<ObjHashable, Obj>>),
    Set(Rc<HashSet<ObjHashable>>),
    FrozenSet(Rc<HashSet<ObjHashable>>),
    // etc.
}

/// This is a f64 wrapper suitable for use as a key in a (Hash)Map, since NaNs compare equal to
/// each other, so it can implement Eq and Hash. `HashF64(-0.0) == HashF64(0.0)`.
#[derive(Clone)]
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

pub struct HashableHashSet<T>(HashSet<T>);
impl<T> Hash for HashableHashSet<T> where T: Hash {
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
impl<T> PartialEq for HashableHashSet<T> where T: Eq + Hash {
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}
impl<T> Eq for HashableHashSet<T> where T: Eq + Hash {}
impl<T> FromIterator<T> for HashableHashSet<T> where T: Eq + Hash {
    fn from_iter<I>(iter: I) -> Self where I: IntoIterator<Item = T> {
        Self(iter.into_iter().collect())
    }
}

#[derive(PartialEq, Eq, Hash, Clone)]
pub enum ObjHashable {
    None,
    StopIteration,
    Ellipsis,
    Bool(bool),
    Long(Rc<BigInt>),
    Float(HashF64),
    Complex(Complex<HashF64>),
    String(Rc<String>),
    Tuple(Rc<Vec<ObjHashable>>),
    FrozenSet(Rc<HashableHashSet<ObjHashable>>),
    // etc.
}

impl TryFrom<&Obj> for ObjHashable {
    type Error = Obj;

    fn try_from(orig: &Obj) -> Result<Self, Obj> {
        match orig {
            Obj::None => Ok(ObjHashable::None),
            Obj::StopIteration => Ok(ObjHashable::StopIteration),
            Obj::Ellipsis => Ok(ObjHashable::Ellipsis),
            Obj::Bool(x) => Ok(ObjHashable::Bool(*x)),
            Obj::Long(x) => Ok(ObjHashable::Long(Rc::clone(x))),
            Obj::Float(x) => Ok(ObjHashable::Float(HashF64(*x))),
            Obj::Complex(Complex { re, im }) => Ok(ObjHashable::Complex(Complex {
                re: HashF64(*re),
                im: HashF64(*im),
            })),
            Obj::String(x) => Ok(ObjHashable::String(Rc::clone(x))),
            Obj::Tuple(x) => Ok(ObjHashable::Tuple(Rc::new(
                x.iter()
                    .map(Self::try_from)
                    .collect::<Result<Vec<Self>, Obj>>()?,
            ))),
            Obj::FrozenSet(x) => Ok(ObjHashable::FrozenSet(Rc::new(
                x.iter()
                    .cloned()
                    .collect::<HashableHashSet<Self>>(),
            ))),
            x => Err(x.clone()),
        }
    }
}

mod utils {
    use num_bigint::{BigUint, Sign};
    use num_traits::Zero;
    use std::cmp::Ordering;

    // TODO: test
    /// Based on _PyLong_AsByteArray in https://github.com/python/cpython/blob/master/Objects/longobject.c
    pub fn biguint_from_pylong_digits(digits: &[u16]) -> BigUint {
        if digits.len() == 0 {
            return BigUint::zero();
        };
        assert!(digits[digits.len() - 1] != 0);
        let mut accum: u64 = 0;
        let mut accumbits: u8 = 0;
        let mut p = Vec::<u32>::new();
        for (i, &thisdigit) in digits.iter().enumerate() {
            accum |= (thisdigit as u64) << accumbits;
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

    pub fn sign_of<T: Ord + Zero>(x: T) -> Sign {
        match x.cmp(&T::zero()) {
            Ordering::Less => Sign::Minus,
            Ordering::Equal => Sign::NoSign,
            Ordering::Greater => Sign::Plus,
        }
    }

    #[cfg(test)]
    mod test {
        use super::*;

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

pub mod read {
    use super::*;
    use num_bigint::BigInt;
    use num_traits::Zero;
    use std::io::{self, Read};
    use std::num::ParseFloatError;
    use std::str::{FromStr, Utf8Error};
    use std::string::FromUtf8Error;

    pub enum Error {
        IoError(io::Error),
        InvalidType(u8),
        RecursionLimitExceeded,
        DigitOutOfRange(u16),
        UnnormalizedLong,
        Utf8Error(Utf8Error),
        FromUtf8Error(FromUtf8Error),
        ParseFloatError(ParseFloatError),
        Null,
        Unhashable(Obj),
    }

    type Result<T> = std::result::Result<T, Error>;

    struct RFile<R: Read> {
        //fp: fs::File,
        depth: Depth,
        readable: R,
        //pos: usize,
        //buf: Option<Vec<u8>>,
        refs: Vec<Obj>,
    }

    macro_rules! define_r {
        ($ident:ident -> $ty:ty; $n:literal) => {
            fn $ident(p: &mut RFile<impl Read>) -> Result<$ty> {
                let mut buf: [u8; $n] = [0; $n];
                p.readable.read_exact(&mut buf).map_err(Error::IoError)?;
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
        p.readable.read_exact(&mut buf).map_err(Error::IoError)?;
        Ok(buf)
    }

    fn r_string(n: usize, p: &mut RFile<impl Read>) -> Result<String> {
        let buf = r_bytes(n, p)?;
        Ok(String::from_utf8(buf).map_err(Error::FromUtf8Error)?)
    }

    fn r_float_str(p: &mut RFile<impl Read>) -> Result<f64> {
        let n = r_byte(p)?;
        let s = r_string(n as usize, p)?;
        Ok(f64::from_str(&s).map_err(Error::ParseFloatError)?)
    }

    // TODO: test
    /// May misbehave on 16-bit platforms.
    fn r_pylong(p: &mut RFile<impl Read>) -> Result<BigInt> {
        let n = r_long(p)? as i32;
        if n == 0 {
            return Ok(BigInt::zero());
        };
        let size = n.wrapping_abs() as u32;
        let mut digits = Vec::<u16>::with_capacity(size as usize);
        for _ in 0..size {
            let d = r_short(p)?;
            if d > (1 << 15) {
                return Err(Error::DigitOutOfRange(d));
            }
            digits.push(d);
        }
        if digits[(size - 1) as usize] == 0 {
            return Err(Error::UnnormalizedLong);
        }
        Ok(BigInt::from_biguint(
            utils::sign_of(n),
            utils::biguint_from_pylong_digits(&digits),
        ))
    }

    fn r_ref(o: &Option<Obj>, p: &mut RFile<impl Read>) {
        if let Some(x) = o {
            p.refs.push(x.clone());
        }
    }

    fn r_vec(n: usize, p: &mut RFile<impl Read>) -> Result<Vec<Obj>> {
        let mut vec = Vec::with_capacity(n);
        for _ in 0..n {
            vec.push(r_object(p)?.ok_or(Error::Null)?);
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
                            ObjHashable::try_from(&key).map_err(Error::Unhashable)?,
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
        for _ in 0..n {
            set.insert(ObjHashable::try_from(&r_object(p)?.ok_or(Error::Null)?).map_err(Error::Unhashable)?);
        }
        Ok(set)
    }

    fn r_object(p: &mut RFile<impl Read>) -> Result<Option<Obj>> {
        let code: u8 = r_byte(p)?;
        let _depth_handle = p
            .depth
            .try_clone()
            .map_or(Err(Error::RecursionLimitExceeded), Ok)?;
        let (flag, type_) = {
            let flag: bool = (code & Type::FLAG_REF) != 0;
            let type_u8: u8 = code & !Type::FLAG_REF;
            let type_: Type =
                Type::from_u8(type_u8).map_or(Err(Error::InvalidType(type_u8)), Ok)?;
            (flag, type_)
        };
        let retval = match type_ {
            Type::Null => None,
            Type::None => Some(Obj::None),
            Type::StopIter => Some(Obj::StopIteration),
            Type::Ellipsis => Some(Obj::Ellipsis),
            Type::False => Some(Obj::Bool(false)),
            Type::True => Some(Obj::Bool(true)),
            Type::Int => Some(Obj::Long(Rc::new(BigInt::from(r_long(p)? as i32)))),
            Type::Int64 => Some(Obj::Long(Rc::new(BigInt::from(r_long64(p)? as i32)))),
            Type::Long => Some(Obj::Long(Rc::new(r_pylong(p)?))),
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
            Type::String | Type::AsciiInterned | Type::Ascii => {
                Some(Obj::String(Rc::new(r_string(r_long(p)? as usize, p)?)))
            }
            Type::ShortAsciiInterned | Type::ShortAscii => {
                Some(Obj::String(Rc::new(r_string(r_byte(p)? as usize, p)?)))
            }
            Type::SmallTuple => Some(Obj::Tuple(Rc::new(r_vec(r_byte(p)? as usize, p)?))),
            Type::Tuple => Some(Obj::Tuple(Rc::new(r_vec(r_long(p)? as usize, p)?))),
            Type::List => Some(Obj::List(Rc::new(r_vec(r_long(p)? as usize, p)?))),
            Type::Dict => Some(Obj::Dict(Rc::new(r_hashmap(p)?))),
            Type::Set => Some(Obj::Set(Rc::new(r_hashset(r_long(p)? as usize, p)?))),
            Type::FrozenSet => Some(Obj::FrozenSet(Rc::new(r_hashset(r_long(p)? as usize, p)?))),
            _ => todo!(), // TODO
        };
        match retval {
            None
            | Some(Obj::None)
            | Some(Obj::StopIteration)
            | Some(Obj::Ellipsis)
            | Some(Obj::Bool(_)) => {}
            Some(_) if flag => r_ref(&retval, p),
            Some(_) => {}
        };
        Ok(retval)
    }

    fn read_object(p: &mut RFile<impl Read>) -> Result<Option<Obj>> {
        r_object(p)
    }

    pub fn marshal_loads(bytes: &[u8]) -> Result<Option<Obj>> {
        let mut rf = RFile {
            depth: Depth::new(),
            readable: bytes,
            //pos: 0,
            //buf: None,
            refs: Vec::<Obj>::new(),
        };
        read_object(&mut rf)
    }
}
