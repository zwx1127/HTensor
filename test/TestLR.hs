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

-- Logistic Regression
predict
    :: (KnownNat m, KnownNat n, Floating a)
    => Vector m a
    -> Matrix n m a
    -> Matrix n 1 a
predict wb x' = let z = x' |*| (cv wb) in (mapTensor sigmoid z)

sigmoid :: Floating a => a -> a
sigmoid x = 1 / (1 + exp (-x))

loss
    :: (KnownNat m, KnownNat n, Floating a)
    => Vector m a
    -> Matrix n m a
    -> Vector n a
    -> a
loss wb x' y =
    let z = x' |*| (cv wb)
    in  (repeatTensor 1)
            |⋅| (((cv y) |⊙| z) |+| (mapTensor (\z' -> log ((exp z')) + 1) z))

-- Gradient Descent
gradient
    :: forall m n a
     . (KnownNat m, KnownNat n, Floating a)
    => Vector m a
    -> Matrix n m a
    -> Vector n a
    -> Vector m a
gradient wb x' y = reshape
    ((transpose x') |*| ((cv y) |-| (mapTensor p (x' |*| (cv wb)))))
  where
    p :: (Num a) => a -> a
    p x = 1 / (1 + (exp (-x)))

gd
    :: forall m n a
     . (KnownNat m, KnownNat n, Floating a)
    => a
    -> Vector m a
    -> Matrix n m a
    -> Vector n a
    -> Vector m a
gd rate wb x' y = wb |+| (rate .*| (gradient wb x' y))

train
    :: forall m n a
     . (KnownNat m, KnownNat n, Floating a)
    => Int
    -> a
    -> Vector m a
    -> Matrix n m a
    -> Vector n a
    -> Vector m a
train i rate wb x' y
    | i == 0    = gd rate wb x' y
    | otherwise = let wb' = train (i - 1) rate wb x' y in gd rate wb' x' y

-- Newton's method
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
            |+| (        (   (1 / ((rv (diagonal (x' |*| x''))) |⋅| p'))
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

-- data set
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

testLR = do
    let x' = (fromArray x) :: Matrix 15 3 Double
    let r  = train 1000 0.5 (fromArray wb) x' (fromArray y)
    print (loss r x' (fromArray y))
    let pr = predict r x'
    print r
    print pr

