{-# LANGUAGE DataKinds #-}
{-# LANGUAGE PolyKinds #-}
{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeOperators #-}

module Data.Tensor.Linear.Seq where

import Data.Tensor.Eval
import Data.Tensor.Linear.Base
import qualified Data.Tensor.Linear.Delay as D
import qualified Data.Tensor.Operators.Seq as ES
import Data.Tensor.Shape
import qualified Data.Tensor.Source.Unbox as UT
import Data.Tensor.Source
import qualified Data.Vector.Unboxed as U
import GHC.TypeLits (KnownNat)

row ::
  ( Source r1 e,
    KnownNat m,
    KnownNat n,
    Num e,
    U.Unbox e
  ) =>
  Matrix r1 m n e ->
  Int ->
  Vector UT.U n e
row arr n = computeS (D.row arr n)

column ::
  ( Source r1 e,
    KnownNat m,
    KnownNat n,
    Num e,
    U.Unbox e
  ) =>
  Matrix r1 m n e ->
  Int ->
  Vector UT.U m e
column arr n = computeS (D.column arr n)

transpose ::
  ( Source r1 e,
    KnownNat m,
    KnownNat n,
    Num e,
    U.Unbox e
  ) =>
  Matrix r1 m n e ->
  Matrix UT.U n m e
transpose arr = computeS $ D.transpose arr

diagonal ::
  ( Source r1 e,
    KnownNat n,
    Num e,
    U.Unbox e
  ) =>
  Matrix r1 n n e ->
  Vector UT.U n e
diagonal arr = computeS $ D.diagonal arr

trace :: (Source r1 e, KnownNat n, Num e, U.Unbox e) => Matrix r1 n n e -> e
trace arr = ES.sumAll $ D.diagonal arr

(.*|) ::
  forall (r1 :: *) sh e.
  ( Source r1 e,
    Shape sh,
    Num e,
    U.Unbox e
  ) =>
  e ->
  Tensor r1 sh e ->
  Tensor UT.U sh e
(.*|) = scalarMul

scalarMul ::
  forall (r1 :: *) sh e.
  ( Source r1 e,
    Shape sh,
    Num e,
    U.Unbox e
  ) =>
  e ->
  Tensor r1 sh e ->
  Tensor UT.U sh e
scalarMul x arr = ES.mapTensor ((*) x) arr

(|*.) ::
  forall (r1 :: *) sh e.
  ( Source r1 e,
    Shape sh,
    Num e,
    U.Unbox e
  ) =>
  Tensor r1 sh e ->
  e ->
  Tensor UT.U sh e
(|*.) = mulScalar

mulScalar ::
  forall (r1 :: *) sh e.
  ( Source r1 e,
    Shape sh,
    Num e,
    U.Unbox e
  ) =>
  Tensor r1 sh e ->
  e ->
  Tensor UT.U sh e
mulScalar arr x = ES.mapTensor ((*) x) arr

(|+|) ::
  ( Source r1 e,
    Source r2 e,
    Shape sh,
    Num e,
    U.Unbox e
  ) =>
  Tensor r1 sh e ->
  Tensor r2 sh e ->
  Tensor UT.U sh e
(|+|) = matplus

matplus ::
  ( Source r1 e,
    Source r2 e,
    Shape sh,
    Num e,
    U.Unbox e
  ) =>
  Tensor r1 sh e ->
  Tensor r2 sh e ->
  Tensor UT.U sh e
matplus arr1 arr2 = computeS $ D.matplus arr1 arr2

(|-|) ::
  ( Source r1 e,
    Source r2 e,
    Shape sh,
    Num e,
    U.Unbox e
  ) =>
  Tensor r1 sh e ->
  Tensor r2 sh e ->
  Tensor UT.U sh e
(|-|) = matminus

matminus ::
  ( Source r1 e,
    Source r2 e,
    Shape sh,
    Num e,
    U.Unbox e
  ) =>
  Tensor r1 sh e ->
  Tensor r2 sh e ->
  Tensor UT.U sh e
matminus arr1 arr2 = computeS $ D.matminus arr1 arr2

(|⊙|) ::
  ( Source r1 e,
    Source r2 e,
    Shape sh,
    Num e,
    U.Unbox e
  ) =>
  Tensor r1 sh e ->
  Tensor r2 sh e ->
  Tensor UT.U sh e
(|⊙|) = hadamard

hadamard ::
  ( Source r1 e,
    Source r2 e,
    Shape sh,
    Num e,
    U.Unbox e
  ) =>
  Tensor r1 sh e ->
  Tensor r2 sh e ->
  Tensor UT.U sh e
hadamard arr1 arr2 = computeS $ D.hadamard arr1 arr2

(|*|) ::
  ( Source r1 e,
    Source r2 e,
    KnownNat m,
    KnownNat p,
    KnownNat n,
    Num e,
    U.Unbox e
  ) =>
  Matrix r1 m p e ->
  Matrix r2 p n e ->
  Matrix UT.U m n e
(|*|) = matmul

matmul ::
  ( Source r1 e,
    Source r2 e,
    KnownNat m,
    KnownNat p,
    KnownNat n,
    Num e,
    U.Unbox e
  ) =>
  Matrix r1 m p e ->
  Matrix r2 p n e ->
  Matrix UT.U m n e
matmul arr1 arr2 = computeS $ D.matmul arr1 arr2

(|⋅|) ::
  forall r1 r2 m e.
  ( Source r1 e,
    Source r2 e,
    KnownNat m,
    Num e,
    U.Unbox e
  ) =>
  RVector r1 m e ->
  CVector r2 m e ->
  e
(|⋅|) = dot

dot ::
  forall r1 r2 m e.
  ( Source r1 e,
    Source r2 e,
    KnownNat m,
    Num e,
    U.Unbox e
  ) =>
  RVector r1 m e ->
  CVector r2 m e ->
  e
dot arr1 arr2 = ES.sumAll $ (row arr1 0) |⊙| (column arr2 0)

normSq :: (Source r a, KnownNat n, U.Unbox a, Num a) => Vector r n a -> a
normSq x = (rv x) |⋅| (cv x)

norm :: (Source r a, KnownNat n, U.Unbox a, Floating a) => Vector r n a -> a
norm x = sqrt $ normSq x