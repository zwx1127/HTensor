{-# LANGUAGE PolyKinds #-}
{-# LANGUAGE KindSignatures #-}
{-# LANGUAGE RankNTypes #-}

module Data.Tensor.Efficient.Operators.Par where

import Data.Tensor.Efficient.Eval
import qualified Data.Tensor.Efficient.Eval.Reduce as R
import Data.Tensor.Efficient.Eval.Target
import qualified Data.Tensor.Efficient.Operators.Delay as D
import Data.Tensor.Efficient.Shape
import Data.Tensor.Efficient.Source
import qualified Data.Vector.Unboxed as U
import System.IO.Unsafe

{-# INLINE [1] foldAll #-}
foldAll :: (Shape sh, Source r a, U.Unbox a, Monad m) => (a -> a -> a) -> a -> Tensor r sh a -> m a
foldAll f z arr =
  let sh = extent arr
      n = size sh
   in return $ unsafePerformIO $ R.foldAllP (\ix -> arr `unsafeIndex` fromIndex sh ix) f z n

{-# INLINE [3] sumAll #-}
sumAll :: (Shape sh, Source r a, U.Unbox a, Num a, Monad m) => Tensor r sh a -> m a
sumAll = foldAll (+) 0

{-# INLINE [1] mapTensor #-}
mapTensor ::
  forall (r1 :: *) (r2 :: *) sh a (m :: * -> *).
  ( Source r1 a,
    Source r2 a,
    Target r2 a,
    Shape sh,
    Monad m
  ) =>
  (a -> a) ->
  Tensor r1 sh a ->
  m (Tensor r2 sh a)
mapTensor f arr = computeP (D.mapTensor f arr)