{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE TypeFamilies #-}

module Data.Tensor.Eval.Target where

import Data.Tensor.Shape
import Data.Tensor.Source

class Target r e where
  data MVec r e

  new :: Int -> IO (MVec r e)

  unsafeWrite :: MVec r e -> Int -> e -> IO ()

  unsafeRead :: MVec r e -> Int -> IO e

  unsafeFreeze :: (Shape sh) => MVec r e -> IO (Tensor r sh e)

  seqMVec :: MVec r e -> a -> a
