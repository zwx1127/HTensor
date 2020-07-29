{-# LANGUAGE PolyKinds #-}
{-# LANGUAGE KindSignatures #-}
{-# LANGUAGE RankNTypes #-}

module Data.Tensor.Operators.Par where

import Data.Tensor.Eval
import qualified Data.Tensor.Eval.Reduce as R
import qualified Data.Tensor.Operators.Delay as D
import Data.Tensor.Shape
import Data.Tensor.Source
import qualified Data.Vector.Unboxed as U
import qualified Data.Tensor.Source.Unbox as UT
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
  forall (r1 :: *) sh a (m :: * -> *).
  ( Source r1 a,
    Shape sh,
    Monad m,
    U.Unbox a
  ) =>
  (a -> a) ->
  Tensor r1 sh a ->
  m (Tensor UT.U sh a)
mapTensor f arr = computeP (D.mapTensor f arr)