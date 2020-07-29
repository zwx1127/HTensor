{-# LANGUAGE DataKinds #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE KindSignatures #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE PolyKinds #-}
{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE TypeOperators #-}

module Neural where

import Data.Tensor.Eval
import Data.Tensor.Linear.Base
import Data.Tensor.Linear.Delay
import Data.Tensor.Operators.Delay
import Data.Tensor.Source
import Data.Tensor.Source.Delay
import Data.Tensor.Source.Unbox
import qualified Data.Vector.Unboxed as U
import GHC.TypeLits
  ( KnownNat,
  )

data Layer r m n a = Layer
  { w :: Matrix r n m a,
    b :: Vector r n a,
    f :: (a -> a),
    f' :: (a -> a)
  }

infixr 5 :~:

data Neural r m n a where
  Neural ::
    ( Source r a,
      KnownNat m,
      KnownNat n,
      Num a
    ) =>
    Layer r m n a ->
    Neural r m n a
  (:~:) ::
    ( Source r a,
      KnownNat m,
      KnownNat n,
      KnownNat o,
      Num a
    ) =>
    Neural r m o a ->
    Neural r o n a ->
    Neural r m n a

forward' ::
  forall r1 r2 a m n p mo.
  ( Source r1 a,
    Source r2 a,
    KnownNat m,
    KnownNat n,
    KnownNat p,
    U.Unbox a,
    Num a,
    Monad mo
  ) =>
  Matrix r1 p m a ->
  Layer r2 m n a ->
  mo (Matrix U p n a)
forward' x Layer {w = _w, b = _b} =
  computeP $
    (x |*| (transpose _w)) |+| (oneDTensor |*| (rv _b))

forward ::
  ( Source r1 a,
    Source r2 a,
    KnownNat m,
    KnownNat n,
    KnownNat p,
    U.Unbox a,
    Num a,
    Monad mo
  ) =>
  Matrix r1 p m a ->
  Layer r2 m n a ->
  mo (Matrix U p n a)
forward x Layer {w = _w, b = _b, f = _f} =
  computeP $ mapTensor _f $ (x |*| (transpose _w)) |+| (oneDTensor |*| (rv _b))

backward ::
  ( Source r1 a,
    Source r2 a,
    Source r3 a,
    KnownNat m,
    KnownNat n,
    KnownNat p,
    U.Unbox a,
    Num a,
    Monad mo
  ) =>
  Matrix r1 p m a ->
  Matrix r2 p n a ->
  Layer r3 m n a ->
  mo (Matrix U p m a)
backward z e Layer {w = _w, b = _b, f = _f, f' = _f'} =
  computeP $ (e |*| _w) |⊙| (mapTensor _f' z)

backwardEnd ::
  forall r1 r2 r3 a m n p mo.
  ( Source r1 a,
    Source r2 a,
    Source r3 a,
    KnownNat m,
    KnownNat n,
    KnownNat p,
    U.Unbox a,
    Num a,
    Monad mo
  ) =>
  Matrix r1 p m a ->
  Matrix r2 p n a ->
  Layer r3 m n a ->
  mo (Matrix U p n a)
backwardEnd x y l@Layer {w = _w, b = _b, f = _f, f' = _f'} = do
  z <- forward' x l
  let a = mapTensor _f z
  computeP $ (a |-| y) |⊙| (mapTensor _f' z)

update ::
  ( Source r1 a,
    Source r2 a,
    KnownNat m,
    KnownNat n,
    KnownNat p,
    U.Unbox a,
    Num a,
    Monad mo
  ) =>
  a ->
  Matrix r1 p m a ->
  Matrix r2 p n a ->
  Layer U m n a ->
  mo (Layer U m n a)
update rate a e l@Layer {w = _w, b = _b} = do
  w' <- computeP $ _w |-| (rate .*| ((transpose e) |*| a))
  b' <- computeP $ _b |-| (reshape ((rate .*| ((transpose e) |*| one))))
  return l {w = w', b = b'}
  where
    one :: (KnownNat p, Num a) => Matrix D p 1 a
    one = oneDTensor

trainLoop ::
  forall r1 r2 a m n p mo.
  ( Source r1 a,
    Source r2 a,
    KnownNat m,
    KnownNat n,
    KnownNat p,
    U.Unbox a,
    Num a,
    Monad mo
  ) =>
  Int ->
  a ->
  Matrix r1 p m a ->
  Matrix r2 p n a ->
  Neural U m n a ->
  mo (Neural U m n a)
trainLoop count rate x y l
  | count == 0 = train rate x y l
  | otherwise = do
    l' <- trainLoop (count - 1) rate x y l
    train rate x y l'

train ::
  forall r1 r2 a m n p mo.
  ( Source r1 a,
    Source r2 a,
    KnownNat m,
    KnownNat n,
    KnownNat p,
    U.Unbox a,
    Num a,
    Monad mo
  ) =>
  a ->
  Matrix r1 p m a ->
  Matrix r2 p n a ->
  Neural U m n a ->
  mo (Neural U m n a)
train rate x y l = do
  (neural, _) <- step rate (x, x) y l
  return neural
  where

step ::
  forall r1 r2 r3 a m n p mo.
  ( Source r1 a,
    Source r2 a,
    Source r3 a,
    KnownNat m,
    KnownNat n,
    KnownNat p,
    U.Unbox a,
    Num a,
    Monad mo
  ) =>
  a ->
  ((Matrix r1 p m a), (Matrix r3 p m a)) ->
  Matrix r2 p n a ->
  Neural U m n a ->
  mo (Neural U m n a, Matrix U p m a)
step _rate (_z, _x) _y (Neural _l) = do
  _e <- backwardEnd _x _y _l
  _e' <- backward _z _e _l
  _l' <- update _rate _x _e _l
  return (Neural _l', _e')
step _rate (_z, _x) _y ((Neural _l) :~: neural) = do
  _z' <- forward' _x _l
  let _x' = mapTensor (f _l) _z'
  (neural', _e) <- step _rate (_z', _x') _y neural
  _e' <- backward _z _e _l
  _l' <- update _rate _x _e _l
  return ((Neural _l') :~: neural', _e')

predict ::
  ( Source r1 a,
    KnownNat m,
    KnownNat n,
    KnownNat p,
    U.Unbox a,
    Num a,
    Monad mo
  ) =>
  Matrix r1 p m a ->
  Neural U m n a ->
  mo (Matrix U p n a)
predict x (Neural l) = forward x l
predict x ((Neural l) :~: neural) = do
  a <- forward x l
  predict a neural
