{-# LANGUAGE DataKinds #-}

module TestBP
    ( testBP
    )
where

import           GHC.TypeLits                   ( KnownNat )
import           Test.Hspec
import           Data.Tensor.Simple.Tensor
import           Data.Tensor.Simple.Linear
import           Data.Tensor.Simple.Neural

sigmoid :: (Floating a) => a -> a
sigmoid x = 1 / (1 + exp (-x))

sigmoid' :: (Floating a) => a -> a
sigmoid' x = (sigmoid x) * (1 - (sigmoid x))

l0 :: Layer 2 3 Float
l0 =
    Layer { w = repeatTensor 1, b = repeatTensor 1, f = sigmoid, f' = sigmoid' }

l1 :: Layer 3 3 Float
l1 =
    Layer { w = repeatTensor 1, b = repeatTensor 1, f = sigmoid, f' = sigmoid' }

l2 :: Layer 3 3 Float
l2 =
    Layer { w = repeatTensor 1, b = repeatTensor 1, f = sigmoid, f' = sigmoid' }

l3 :: Layer 3 1 Float
l3 =
    Layer { w = repeatTensor 1, b = repeatTensor 1, f = sigmoid, f' = sigmoid' }

model = Neural l0 :~: Neural l3
x =
    [ 0.697
    , 0.460
    , 0.774
    , 0.376
    , 0.634
    , 0.264
    , 0.608
    , 0.318
    , 0.556
    , 0.215
    , 0.403
    , 0.237
    , 0.481
    , 0.149
    , 0.437
    , 0.211
    , 0.666
    , 0.091
    , 0.243
    , 0.267
    , 0.245
    , 0.057
    , 0.343
    , 0.099
    , 0.639
    , 0.161
    , 0.657
    , 0.198
    , 0.360
    , 0.370
    , 0.593
    , 0.042
    , 0.719
    , 0.103
    ]
y = (take 8 (repeat 1.0)) <> (take 9 (repeat 0.0))
testBP = do
    let x' = (fromArray x) :: Matrix 17 2 Float
    let y' = (fromArray y) :: Matrix 17 1 Float
    let r  = trainLoop 200000 0.05 x' y' model
    let pr = predict x' r
    print pr
