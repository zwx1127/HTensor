{-# LANGUAGE DataKinds #-}

module TestBP
  ( testBP,
  )
where

import Data.Tensor.Linear.Base
import Data.Tensor.Source
import Data.Tensor.Source.Unbox
import Neural

sigmoid :: (Floating a) => a -> a
sigmoid x = 1 / (1 + exp (- x))

sigmoid' :: (Floating a) => a -> a
sigmoid' x = (sigmoid x) * (1 - (sigmoid x))

l0 :: Layer U 2 10 Float
l0 =
  Layer {w = repeatTensor 1, b = repeatTensor 1, f = sigmoid, f' = sigmoid'}

l1 :: Layer U 10 1 Float
l1 =
  Layer {w = repeatTensor 1, b = repeatTensor 1, f = sigmoid, f' = sigmoid'}

model :: Neural U 2 1 Float
model = Neural l0 :~: Neural l1

trainX :: [Float]
trainX =
  [ 0.697,
    0.460,
    0.774,
    0.376,
    0.634,
    0.264,
    0.608,
    0.318,
    0.556,
    0.215,
    0.403,
    0.237,
    0.481,
    0.149,
    0.437,
    0.211,
    0.666,
    0.091,
    0.243,
    0.267,
    0.245,
    0.057,
    0.343,
    0.099,
    0.639,
    0.161,
    0.657,
    0.198,
    0.360,
    0.370,
    0.593,
    0.042,
    0.719,
    0.103
  ]

trainY :: [Float]
trainY = (take 8 (repeat 1.0)) <> (take 9 (repeat 0.0))

testBP :: IO ()
testBP = do
  let x' = (fromList trainX) :: Matrix U 17 2 Float
  let y' = (fromList trainY) :: Matrix U 17 1 Float
  r <- trainLoop 500000 0.1 x' y' model
  pr <- predict x' r
  print pr
