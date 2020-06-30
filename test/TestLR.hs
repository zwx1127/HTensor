{-# LANGUAGE DataKinds #-}
{-# LANGUAGE PolyKinds #-}
{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeOperators #-}

module TestLR
    ( testLR
    )
where

import           Test.Hspec
import           Tensor
import           Linear
import           Data.Proxy                     ( Proxy(..) )
import           GHC.TypeLits                   ( KnownNat
                                                , natVal
                                                )

predict :: (KnownNat m, Floating a) => Vector m a -> Vector m a -> a
predict wb x' = let z = (rv wb) |.| (cv x') in (exp z) / (1 + exp z)

sigmoid :: (KnownNat m, Floating a) => Vector m a -> Vector m a -> a
sigmoid wb x' = let z = (rv wb) |.| (cv x') in 1 / (1 + exp (-z))

loss :: (KnownNat m, Floating a) => Vector m a -> Vector m a -> a -> a
loss wb x' y = let z = (rv wb) |.| (cv x') in y * z + log ((exp z) + 1)

newton
    :: forall m n a
     . (KnownNat m, KnownNat n, Floating a)
    => Vector m a
    -> Matrix n m a
    -> Vector n a
    -> Vector m a
newton wb x' y =
    let p   = (mapTensor (\z -> (exp z) / (1 + exp z)) (x' |*| (cv wb)))
        p'  = mapTensor (\x -> x * (1 - x)) p
        x'' = transpose x'
    in  wb
            |+| (        (   (1 / ((rv (diagonal (x' |*| x''))) |.| p'))
                         .*| (x'' |*| ((cv y) |-| p))
                         )
                `column` 0
                )

trainNewton
    :: (KnownNat m, KnownNat n, Floating a)
    => Int
    -> Vector m a
    -> Matrix n m a
    -> Vector n a
    -> Vector m a
trainNewton i wb x' y
    | i == 0    = newton wb x' y
    | otherwise = let wb' = trainNewton (i - 1) wb x' y in newton wb' x' y

x =
    [ 0.697
    , 0.460
    , 1
    , 0.774
    , 0.376
    , 1
    , 0.634
    , 0.264
    , 1
    , 0.608
    , 0.318
    , 1
    , 0.556
    , 0.215
    , 1
    , 0.403
    , 0.237
    , 1
    , 0.481
    , 0.149
    , 1
    , 0.437
    , 0.211
    , 1
    , 0.666
    , 0.091
    , 1
    , 0.243
    , 0.267
    , 1
    , 0.245
    , 0.057
    , 1
    , 0.343
    , 0.099
    , 1
    , 0.639
    , 0.161
    , 1
    , 0.657
    , 0.198
    , 1
    , 0.360
    , 0.370
    , 1
    , 0.593
    , 0.042
    , 1
    , 0.719
    , 0.103
    , 1
    ]

y = (take 8 (repeat 1.0)) <> (take 9 (repeat 0.0))

wb = [0, 0, 0]

positive = [0.697, 0.460, 1]

negative = [0.360, 0.370, 1]

testLR = do
    let x' = (fromArray x) :: Matrix 15 3 Double
    let r  = trainNewton 5000 (fromArray wb) x' (fromArray y)
    let pr = predict r (fromArray positive)
    let nr = predict r (fromArray negative)
    print r
    print pr
    print nr

