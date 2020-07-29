{-# LANGUAGE BangPatterns #-}
{-# LANGUAGE MagicHash #-}

module Data.Tensor.Operators.Seq where

import Data.Tensor.Eval
import qualified Data.Tensor.Eval.Reduce as R
import qualified Data.Tensor.Operators.Delay as D
import Data.Tensor.Shape
import qualified Data.Tensor.Source.Unbox as UT
import Data.Tensor.Source
import qualified Data.Vector.Unboxed as U
import GHC.Exts

{-# INLINE [1] foldAll #-}
foldAll :: (Shape sh, Source r a) => (a -> a -> a) -> a -> Tensor r sh a -> a
foldAll f z arr =
  let !ex = extent arr
      !(I# n) = size ex
   in R.foldAllS (\ix -> arr `unsafeIndex` fromIndex ex (I# ix)) f z n

{-# INLINE [3] sumAll #-}
sumAll :: (Shape sh, Source r a, Num a) => Tensor r sh a -> a
sumAll = foldAll (+) 0

{-# INLINE [1] mapTensor #-}
mapTensor ::
  ( Source r1 a,
    Shape sh,
    U.Unbox a
  ) =>
  (a -> a) ->
  Tensor r1 sh a ->
  Tensor UT.U sh a
mapTensor f arr = computeS (D.mapTensor f arr)