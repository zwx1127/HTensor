module Main where

import Test.Hspec
-- import TestBP
-- import TestEF
import TestLR

spec :: Spec
spec = do
  it "test Newton train" $ do
    testLR
-- it "test bp neural network" $ do
--     testBP
-- it "test efficitent" $ do
--   testEF


main :: IO ()
main = hspec spec
