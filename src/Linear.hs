{-# LANGUAGE DataKinds #-}
{-# LANGUAGE PolyKinds #-}
{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeOperators #-}

module Linear where

import           Tensor
import           Data.Proxy                     ( Proxy(..) )
import           GHC.TypeLits                   ( KnownNat
                                                , natVal
                                                )

type Matrix m n a = Tensor (m >< n) a

type Vector m a = Tensor m a

type RVector m a = Matrix 1 m a

type CVector m a = Matrix m 1 a

rv :: (KnownNat m, Num a) => Vector m a -> RVector m a
rv (Tensor xs) = fromArray xs

cv :: (KnownNat m, Num a) => Vector m a -> CVector m a
cv (Tensor xs) = fromArray xs

rv2cv :: (KnownNat m, Num a) => RVector m a -> CVector m a
rv2cv (Tensor xs) = fromArray xs

cv2rv :: (KnownNat m, Num a) => CVector m a -> RVector m a
cv2rv (Tensor xs) = fromArray xs

row
  :: forall m n a
   . (KnownNat m, KnownNat n, Num a)
  => Matrix m n a
  -> Int
  -> Vector n a
row (Tensor xs) rn = fromArray
  [ xs !! i
  | i <-
    (index (fromInteger (natVal (Proxy @m)))
           (fromInteger (natVal (Proxy @n)))
           rn
    )
  ]
 where
  index :: Int -> Int -> Int -> [Int]
  index r c rn | rn < r && rn >= 0 = [(rn * c) .. ((rn + 1) * c - 1)]
               | otherwise         = error "index not match shap."

column
  :: forall m n a
   . (KnownNat m, KnownNat n, Num a)
  => Matrix m n a
  -> Int
  -> Vector m a
column (Tensor xs) cn = fromArray
  [ xs !! i
  | i <-
    (index (fromInteger (natVal (Proxy @m)))
           (fromInteger (natVal (Proxy @n)))
           cn
    )
  ]
 where
  index :: Int -> Int -> Int -> [Int]
  index r c cn | cn < c && cn >= 0 = [ i * c + cn | i <- [0 .. r - 1] ]
               | otherwise         = error "index not match shap."

transpose
  :: forall m n a
   . (KnownNat m, KnownNat n, Num a)
  => Matrix m n a
  -> Matrix n m a
transpose (Tensor xs) = fromArray
  [ xs !! i
  | i <-
    (trIndex (fromInteger (natVal (Proxy @m))) (fromInteger (natVal (Proxy @n)))
    )
  ]
 where
  trIndex :: Int -> Int -> [Int]
  trIndex r c = [ x * c + y | y <- [0 .. c - 1], x <- [0 .. r - 1] ]

trace :: forall n a . (KnownNat n, Num a) => Matrix n n a -> a
trace (Tensor xs) = sum
  [ xs !! i
  | i <-
    (index (fromInteger (natVal (Proxy @n))) (fromInteger (natVal (Proxy @n))))
  ]
 where
  index :: Int -> Int -> [Int]
  index n c | n == 1    = [0]
            | otherwise = ((c + 1) * (n - 1)) : (index (n - 1) c)

diagonal :: forall n a . (KnownNat n, Num a) => Matrix n n a -> Vector n a
diagonal (Tensor xs) = fromArray
  [ xs !! i
  | i <-
    (index (fromInteger (natVal (Proxy @n))) (fromInteger (natVal (Proxy @n))))
  ]
 where
  index :: Int -> Int -> [Int]
  index n c | n == 1    = [0]
            | otherwise = ((c + 1) * (n - 1)) : (index (n - 1) c)
(.*|) :: (HasDim d, Num a) => a -> Tensor d a -> Tensor d a
(.*|) = scalarMul

scalarMul :: (HasDim d, Num a) => a -> Tensor d a -> Tensor d a
scalarMul s (Tensor xs) = fromArray (fmap ((*) s) xs)

(|*.) :: (HasDim d, Num a) => Tensor d a -> a -> Tensor d a
(|*.) = mulScalar

mulScalar :: (HasDim d, Num a) => Tensor d a -> a -> Tensor d a
mulScalar (Tensor xs) s = fromArray (fmap ((*) s) xs)

(|+|) :: (HasDim d, Num a) => Tensor d a -> Tensor d a -> Tensor d a
(|+|) = matplus

matplus :: (HasDim d, Num a) => Tensor d a -> Tensor d a -> Tensor d a
matplus (Tensor xs1) (Tensor xs2) = fromArray (pt xs1 xs2)
 where
  pt :: (Num a) => [a] -> [a] -> [a]
  pt []         []         = []
  pt (x1 : xs1) (x2 : xs2) = (x1 + x2) : (pt xs1 xs2)
  pt _          _          = error "index not match shap."

(|-|) :: (HasDim d, Num a) => Tensor d a -> Tensor d a -> Tensor d a
(|-|) = matminus

matminus :: (HasDim d, Num a) => Tensor d a -> Tensor d a -> Tensor d a
matminus (Tensor xs1) (Tensor xs2) = fromArray (pt xs1 xs2)
 where
  pt :: (Num a) => [a] -> [a] -> [a]
  pt []         []         = []
  pt (x1 : xs1) (x2 : xs2) = (x1 - x2) : (pt xs1 xs2)
  pt _          _          = error "index not match shap."

(|⋅|) :: (KnownNat m, Num a) => RVector m a -> CVector m a -> a
(|⋅|) = dot

dot :: (KnownNat m, Num a) => RVector m a -> CVector m a -> a
dot rv cv = (rv |*| cv) !!! [0, 0]

(|*|)
  :: forall m p n a
   . (KnownNat m, KnownNat p, KnownNat n, Num a)
  => Matrix m p a
  -> Matrix p n a
  -> Matrix m n a
(|*|) = matmul

matmul
  :: forall m p n a
   . (KnownNat m, KnownNat p, KnownNat n, Num a)
  => Matrix m p a
  -> Matrix p n a
  -> Matrix m n a
matmul (Tensor xs1) (Tensor xs2) = fromArray
  (mm (fromInteger (natVal (Proxy @m)))
      (fromInteger (natVal (Proxy @p)))
      xs1
      (fromInteger (natVal (Proxy @p)))
      (fromInteger (natVal (Proxy @n)))
      xs2
  )
 where
  mrc :: (Num a) => [a] -> [a] -> a
  mrc []         []         = 0
  mrc (x1 : xs1) (x2 : xs2) = x1 * x2 + mrc xs1 xs2
  mrc _          _          = error "index not match shap."
  mm :: (Num a) => Int -> Int -> [a] -> Int -> Int -> [a] -> [a]
  mm m1 n1 xs1 m2 n2 xs2 =
    [ (mrc ([ xs1 !! i1 | i1 <- [(x * n1) .. ((x + 1) * n1 - 1)] ])
           ([ xs2 !! i2 | i2 <- [ i * n2 + y | i <- [0 .. m2 - 1] ] ])
      )
    | x <- [0 .. m1 - 1]
    , y <- [0 .. n2 - 1]
    ]

(|⊙|)
  :: (KnownNat m, KnownNat n, Num a)
  => Matrix m n a
  -> Matrix m n a
  -> Matrix m n a
(|⊙|) = hadamard

hadamard
  :: (KnownNat m, KnownNat n, Num a)
  => Matrix m n a
  -> Matrix m n a
  -> Matrix m n a
hadamard (Tensor xs1) (Tensor xs2) = fromArray (p xs1 xs2)
 where
  p :: (Num a) => [a] -> [a] -> [a]
  p (x1 : [] ) (x2 : [] ) = (x1 * x2) : []
  p (x1 : xs1) (x2 : xs2) = (x1 * x2) : (p xs1 xs2)
  p _          _          = error "index not match shap."
