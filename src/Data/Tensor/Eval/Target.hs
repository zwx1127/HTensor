{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE TypeFamilies #-}

module Data.Tensor.Eval.Target where

import           Data.Tensor.Source
import           Data.Tensor.Shape

class Target r e where

    data MVec r e

    new :: Int -> IO (MVec r e)

    unsafeWrite :: MVec r e -> Int -> e -> IO ()

    unsafeFreeze ::(Shape sh) => MVec r e -> IO (Tensor r sh e)

    seqMVec :: MVec r e -> a -> a
