{-# LANGUAGE DataKinds #-}
{-# LANGUAGE PolyKinds #-}
{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeOperators #-}

module Data.Tensor.Efficient.Linear.Delay where

import Data.Tensor.Efficient.Linear.Base
import qualified Data.Tensor.Efficient.Operators.Delay as D
import qualified Data.Tensor.Efficient.Operators.Seq as S
import Data.Tensor.Efficient.Shape
import Data.Tensor.Efficient.Source
import Data.Tensor.Efficient.Source.Delay
import GHC.TypeLits (KnownNat)

row ::
  forall m n r e.
  ( Source r e,
    KnownNat m,
    KnownNat n,
    Num e
  ) =>
  Matrix r m n e ->
  Int ->
  Vector D n e
row arr n = generate (gen arr n)
  where
    cast x (y :. Z) = x :. y :. Z
    gen a x = (\ix -> a !? (cast x ix))

column ::
  forall m n r e.
  ( Source r e,
    KnownNat m,
    KnownNat n,
    Num e
  ) =>
  Matrix r m n e ->
  Int ->
  Vector D m e
column arr n = generate (gen arr n)
  where
    cast y (x :. Z) = x :. y :. Z
    gen a x = (\ix -> a !? (cast x ix))

transpose :: (Source r e, KnownNat m, KnownNat n, Num e) => Matrix r m n e -> Matrix D n m e
transpose arr = generate (gen arr)
  where
    gen a = (\(x :. y :. Z) -> a !? (y :. x :. Z))

diagonal :: (Source r e, KnownNat n, Num e) => Matrix r n n e -> Vector D n e
diagonal arr = generate (gen arr)
  where
    gen a = (\(x :. Z) -> a !? (x :. x :. Z))

(.*|) :: forall (r :: *) e sh. (Source r e, Shape sh, Num e) => e -> Tensor r sh e -> Tensor D sh e
(.*|) = scalarMul

scalarMul :: forall (r :: *) e sh. (Source r e, Shape sh, Num e) => e -> Tensor r sh e -> Tensor D sh e
scalarMul x arr = D.mapTensor ((*) x) arr

(|*.) :: forall (r :: *) e sh. (Source r e, Shape sh, Num e) => Tensor r sh e -> e -> Tensor D sh e
(|*.) = mulScalar

mulScalar :: forall (r :: *) e sh. (Source r e, Shape sh, Num e) => Tensor r sh e -> e -> Tensor D sh e
mulScalar arr x = D.mapTensor ((*) x) arr

(|+|) :: (Source r1 e, Source r2 e, Shape sh, Num e) => Tensor r1 sh e -> Tensor r2 sh e -> Tensor D sh e
(|+|) = matplus

matplus :: (Source r1 e, Source r2 e, Shape sh, Num e) => Tensor r1 sh e -> Tensor r2 sh e -> Tensor D sh e
matplus arr1 arr2 = generate (\ix -> (arr1 !? ix) + (arr2 !? ix))

(|-|) :: (Source r1 e, Source r2 e, Shape sh, Num e) => Tensor r1 sh e -> Tensor r2 sh e -> Tensor D sh e
(|-|) = matminus

matminus :: (Source r1 e, Source r2 e, Shape sh, Num e) => Tensor r1 sh e -> Tensor r2 sh e -> Tensor D sh e
matminus arr1 arr2 = generate (\ix -> (arr1 !? ix) - (arr2 !? ix))

(|⊙|) :: (Source r1 e, Source r2 e, Shape sh, Num e) => Tensor r1 sh e -> Tensor r2 sh e -> Tensor D sh e
(|⊙|) = hadamard

hadamard :: (Source r1 e, Source r2 e, Shape sh, Num e) => Tensor r1 sh e -> Tensor r2 sh e -> Tensor D sh e
hadamard arr1 arr2 = generate (\ix -> (arr1 !? ix) * (arr2 !? ix))

(|*|) ::
  ( Source r1 e,
    Source r2 e,
    KnownNat m,
    KnownNat p,
    KnownNat n,
    Num e
  ) =>
  Matrix r1 m p e ->
  Matrix r2 p n e ->
  Matrix D m n e
(|*|) = matmul

matmul ::
  ( Source r1 e,
    Source r2 e,
    KnownNat m,
    KnownNat p,
    KnownNat n,
    Num e
  ) =>
  Matrix r1 m p e ->
  Matrix r2 p n e ->
  Matrix D m n e
matmul arr1 arr2 = generate (\(x :. y :. Z) -> S.sumAll $ (row arr1 x) |⊙| (column arr2 y))