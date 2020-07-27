{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE DataKinds #-}

module TestEF
  ( testEF,
  )
where

import Data.Tensor.Efficient.Linear.Base
import Data.Tensor.Efficient.Linear.Par
import Data.Tensor.Efficient.Source
import Data.Tensor.Efficient.Source.Unbox

t :: Matrix U 3 4 Int
t = fromList [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]

testEF :: IO ()
testEF = do
  -- let t' = computeS (_row t 1) :: Vector V 4 Int
  t' <- transpose t
  t'' <- t |*| t'
  print t''
