module Main where

import Test.Hspec
-- import TestBP
-- import TestEF
import TestLR
import TestPLR

spec :: Spec
spec = do
  -- it "test Newton train" $ do
  --   testLR
  it "test P Newton train" $ do
    testPLR

-- it "test bp neural network" $ do
--     testBP
-- it "test efficitent" $ do
--   testEF

main :: IO ()
main = hspec spec
