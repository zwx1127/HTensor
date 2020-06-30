{-# LANGUAGE DataKinds #-}

module TestBP
    ( testBP
    )
where

import           Test.Hspec
import           Tensor
import           Neural

sigmoid :: (Floating a) => a -> a
sigmoid x = 1 / (1 + exp (-x))

l0 :: Dense 3 2 Float
l0 =
    Dense { w = fromArray [0 .. 5], x = fromArray [0 .. 2], b = 0, f = sigmoid }

l1 :: Dense 2 1 Float
l1 =
    Dense { w = fromArray [0 .. 1], x = fromArray [0 .. 1], b = 1, f = sigmoid }

l2 :: Dense 1 3 Float
l2 = Dense { w = fromArray [0 .. 2], x = fromArray [0], b = 2, f = sigmoid }

testBP = print "test bp"

spec :: Spec
spec = do
    it "test bp train" $ do
        testBP
