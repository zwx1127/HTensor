{-# LANGUAGE DataKinds #-}
{-# LANGUAGE PolyKinds #-}
{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeOperators #-}

module Data.Tensor.Linear.Par where

import Data.Tensor.Eval
import Data.Tensor.Linear.Base
import qualified Data.Tensor.Linear.Delay as D
import qualified Data.Tensor.Operators.Par as EP
import Data.Tensor.Shape
import Data.Tensor.Source
import qualified Data.Tensor.Source.Unbox as UT
import qualified Data.Vector.Unboxed as U
import GHC.TypeLits (KnownNat)

row ::
  ( Source r1 e,
    KnownNat m,
    KnownNat n,
    Num e,
    U.Unbox e,
    Monad mo
  ) =>
  Matrix r1 m n e ->
  Int ->
  mo (Vector UT.U n e)
row arr n = computeP (D.row arr n)

column ::
  ( Source r1 e,
    KnownNat m,
    KnownNat n,
    Num e,
    U.Unbox e,
    Monad mo
  ) =>
  Matrix r1 m n e ->
  Int ->
  mo (Vector UT.U m e)
column arr n = computeP $ D.column arr n

transpose ::
  ( Source r1 e,
    KnownNat m,
    KnownNat n,
    Num e,
    U.Unbox e,
    Monad mo
  ) =>
  Matrix r1 m n e ->
  mo (Matrix UT.U n m e)
transpose arr = computeP $ D.transpose arr

diagonal ::
  ( Source r1 e,
    KnownNat n,
    Num e,
    U.Unbox e,
    Monad mo
  ) =>
  Matrix r1 n n e ->
  mo (Vector UT.U n e)
diagonal arr = computeP $ D.diagonal arr

trace ::
  ( Source r1 e,
    KnownNat n,
    Num e,
    U.Unbox e,
    Monad mo
  ) =>
  Matrix r1 n n e ->
  mo e
trace arr = EP.sumAll $ D.diagonal arr

(.*|) ::
  forall (r1 :: *) sh e (mo :: * -> *).
  ( Source r1 e,
    Shape sh,
    Num e,
    U.Unbox e,
    Monad mo
  ) =>
  e ->
  Tensor r1 sh e ->
  mo (Tensor UT.U sh e)
(.*|) = scalarMul

scalarMul ::
  forall (r1 :: *) sh e (mo :: * -> *).
  ( Source r1 e,
    Shape sh,
    Num e,
    U.Unbox e,
    Monad mo
  ) =>
  e ->
  Tensor r1 sh e ->
  mo (Tensor UT.U sh e)
scalarMul x arr = EP.mapTensor ((*) x) arr

(|*.) ::
  forall (r1 :: *) sh e (mo :: * -> *).
  ( Source r1 e,
    Shape sh,
    Num e,
    U.Unbox e,
    Monad mo
  ) =>
  Tensor r1 sh e ->
  e ->
  mo (Tensor UT.U sh e)
(|*.) = mulScalar

mulScalar ::
  forall (r1 :: *) sh e (mo :: * -> *).
  ( Source r1 e,
    Shape sh,
    Num e,
    U.Unbox e,
    Monad mo
  ) =>
  Tensor r1 sh e ->
  e ->
  mo (Tensor UT.U sh e)
mulScalar arr x = EP.mapTensor ((*) x) arr

(|+|) ::
  ( Source r1 e,
    Source r2 e,
    Shape sh,
    Num e,
    U.Unbox e,
    Monad mo
  ) =>
  Tensor r1 sh e ->
  Tensor r2 sh e ->
  mo (Tensor UT.U sh e)
(|+|) = matplus

matplus ::
  ( Source r1 e,
    Source r2 e,
    Shape sh,
    Num e,
    U.Unbox e,
    Monad mo
  ) =>
  Tensor r1 sh e ->
  Tensor r2 sh e ->
  mo (Tensor UT.U sh e)
matplus arr1 arr2 = computeP $ D.matplus arr1 arr2

(|-|) ::
  ( Source r1 e,
    Source r2 e,
    Shape sh,
    Num e,
    U.Unbox e,
    Monad mo
  ) =>
  Tensor r1 sh e ->
  Tensor r2 sh e ->
  mo (Tensor UT.U sh e)
(|-|) = matminus

matminus ::
  ( Source r1 e,
    Source r2 e,
    Shape sh,
    Num e,
    U.Unbox e,
    Monad mo
  ) =>
  Tensor r1 sh e ->
  Tensor r2 sh e ->
  mo (Tensor UT.U sh e)
matminus arr1 arr2 = computeP $ D.matminus arr1 arr2

(|⊙|) ::
  ( Source r1 e,
    Source r2 e,
    Shape sh,
    Num e,
    U.Unbox e,
    Monad mo
  ) =>
  Tensor r1 sh e ->
  Tensor r2 sh e ->
  mo (Tensor UT.U sh e)
(|⊙|) = hadamard

hadamard ::
  ( Source r1 e,
    Source r2 e,
    Shape sh,
    Num e,
    U.Unbox e,
    Monad mo
  ) =>
  Tensor r1 sh e ->
  Tensor r2 sh e ->
  mo (Tensor UT.U sh e)
hadamard arr1 arr2 = computeP $ D.hadamard arr1 arr2

(|*|) ::
  ( Source r1 e,
    Source r2 e,
    KnownNat m,
    KnownNat p,
    KnownNat n,
    Num e,
    U.Unbox e,
    Monad mo
  ) =>
  Matrix r1 m p e ->
  Matrix r2 p n e ->
  mo (Matrix UT.U m n e)
(|*|) = matmul

matmul ::
  ( Source r1 e,
    Source r2 e,
    KnownNat m,
    KnownNat p,
    KnownNat n,
    Num e,
    U.Unbox e,
    Monad mo
  ) =>
  Matrix r1 m p e ->
  Matrix r2 p n e ->
  mo (Matrix UT.U m n e)
matmul arr1 arr2 = computeP $ D.matmul arr1 arr2

(|⋅|) ::
  forall r1 r2 m e mo.
  ( Source r1 e,
    Source r2 e,
    KnownNat m,
    Num e,
    U.Unbox e,
    Monad mo
  ) =>
  RVector r1 m e ->
  CVector r2 m e ->
  mo e
(|⋅|) = dot

dot ::
  forall r1 r2 m e mo.
  ( Source r1 e,
    Source r2 e,
    KnownNat m,
    Num e,
    U.Unbox e,
    Monad mo
  ) =>
  RVector r1 m e ->
  CVector r2 m e ->
  mo e
dot arr1 arr2 = do
  let r = D.row arr1 1
  let c = D.column arr2 1
  EP.sumAll $ r D.|⊙| c