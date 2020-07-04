{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE PolyKinds #-}
{-# LANGUAGE KindSignatures #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE TypeApplications #-}

module Neural where

import           Data.Proxy                     ( Proxy(..) )
import           GHC.TypeLits                   ( KnownNat
                                                , natVal
                                                )
import           Tensor
import           Linear

data Layer m n a = Layer {
    w :: Matrix n m a,
    b :: Vector n a,
    f :: (a -> a),
    f' :: (a -> a)
}

infixr 5 :~:

data Neural m n a where
    Neural ::(KnownNat m, KnownNat n, Num a) => Layer m n a -> Neural m n a
    (:~:) ::(KnownNat m, KnownNat n, KnownNat o, Num a) => Neural m o a -> Neural o n a -> Neural m n a

forward'
    :: (KnownNat m, KnownNat n, KnownNat p, Num a)
    => Matrix p m a
    -> Layer m n a
    -> Matrix p n a
forward' x Layer { w = w, b = b } =
    (x |*| (transpose w)) |+| ((repeatTensor 1) |*| (rv b))

forward
    :: (KnownNat m, KnownNat n, KnownNat p, Num a)
    => Matrix p m a
    -> Layer m n a
    -> Matrix p n a
forward x Layer { w = w, b = b, f = f } =
    mapTensor f ((x |*| (transpose w)) |+| ((repeatTensor 1) |*| (rv b)))


backward
    :: (KnownNat m, KnownNat n, KnownNat p, Num a)
    => Matrix p m a
    -> Matrix p n a
    -> Layer m n a
    -> Matrix p m a
backward z e Layer { w = w, b = b, f = f, f' = f' } =
    (e |*| w) |⊙| (mapTensor f' z)

backwardEnd
    :: (KnownNat m, KnownNat n, KnownNat p, Num a)
    => Matrix p m a
    -> Matrix p n a
    -> Layer m n a
    -> Matrix p n a
backwardEnd x y l@Layer { w = w, b = b, f = f, f' = f' } =
    let z = forward' x l
        a = mapTensor f z
    in  (a |-| y) |⊙| (mapTensor f' z)

update
    :: (KnownNat m, KnownNat n, KnownNat p, Num a)
    => a
    -> Matrix p m a
    -> Matrix p n a
    -> Layer m n a
    -> Layer m n a
update rate a e l@Layer { w = w, b = b } =
    let w' = w |-| (rate .*| ((transpose e) |*| a))
        b' = b |-| (reshape ((rate .*| ((transpose e) |*| repeat1))))
    in  l { w = w', b = b' }
  where
    repeat1 :: (KnownNat p, Num a) => Matrix p 1 a
    repeat1 = repeatTensor 1

trainLoop
    :: (KnownNat m, KnownNat n, KnownNat p, Num a)
    => Int
    -> a
    -> Matrix p m a
    -> Matrix p n a
    -> Neural m n a
    -> Neural m n a
trainLoop count rate x y l
    | count == 0 = train rate x y l
    | otherwise = let l' = trainLoop (count - 1) rate x y l
                  in  train rate x y l'

train
    :: (KnownNat m, KnownNat n, KnownNat p, Num a)
    => a
    -> Matrix p m a
    -> Matrix p n a
    -> Neural m n a
    -> Neural m n a
train rate x y l = let (neural, _) = step rate (x, x) y l in neural
  where
    step
        :: (KnownNat m, KnownNat n, KnownNat p, Num a)
        => a
        -> ((Matrix p m a), (Matrix p m a))
        -> Matrix p n a
        -> Neural m n a
        -> (Neural m n a, Matrix p m a)
    step rate (z, x) y (Neural l) =
        let e  = backwardEnd x y l
            e' = backward z e l
        in  (Neural (update rate x e l), e')
    step rate (z, x) y ((Neural l) :~: neural) =
        let z'           = forward' x l
            x'           = mapTensor (f l) z'
            (neural', e) = step rate (z', x') y neural
            e'           = backward z e l
            l'           = update rate x e l
        in  ((Neural l') :~: neural', e')

predict
    :: (KnownNat m, KnownNat n, KnownNat p, Num a)
    => Matrix p m a
    -> Neural m n a
    -> Matrix p n a
predict x (Neural l             ) = forward x l
predict x ((Neural l) :~: neural) = let a = forward x l in predict a neural
