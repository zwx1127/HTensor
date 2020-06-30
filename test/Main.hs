module Main where

import           Test.Hspec
import           TestLR
import           TestBP

spec :: Spec
spec = do
    it "test Newton train" $ do
        testLR
    it "test bp neural network" $ do
        testBP


main :: IO ()
main = hspec spec
