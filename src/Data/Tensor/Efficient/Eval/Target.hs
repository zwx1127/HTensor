{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE TypeFamilies #-}

module Data.Tensor.Efficient.Eval.Target where

import           Data.Tensor.Efficient.Source
import           Data.Tensor.Efficient.Shape

class Target r e where

    data MVec r e

    new :: Int -> IO (MVec r e)

    unsafeWrite :: MVec r e -> Int -> e -> IO ()

    unsafeFreeze ::(Shape sh) => MVec r e -> IO (Tensor r sh e)

    seqMVec :: MVec r e -> a -> a
