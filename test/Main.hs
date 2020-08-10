module Main where

import SVM
import Test.Hspec
import TestBP
import TestEF
import TestLR

spec :: Spec
spec = do
  it "LR" $ do
    testLR
  it "BP" $ do
    testBP
  it "EF" $ do
    testEF
  it "SVM" $ do
    testSVM

main :: IO ()
main = hspec spec
