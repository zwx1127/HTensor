{-# LANGUAGE MonoLocalBinds #-}
{-# LANGUAGE FlexibleContexts #-}

module Data.Tensor.Efficient.Eval where

import Data.Tensor.Efficient.Eval.Load
import Data.Tensor.Efficient.Eval.Target
import Data.Tensor.Efficient.Shape
import Data.Tensor.Efficient.Source
import Data.Tensor.Efficient.Source.Delay
import System.IO.Unsafe

{-# INLINE [4] now #-}
now :: (Shape sh, Source r e, Monad m) => Tensor r sh e -> m (Tensor r sh e)
now arr = do
  arr `seqTensor` return ()
  return arr

{-# INLINE [4] suspendedComputeP #-}
suspendedComputeP :: (Load r1 sh e, Target r2 e) => Tensor r1 sh e -> Tensor r2 sh e
suspendedComputeP arr1 = arr1 `seqTensor` unsafePerformIO $ do
  mvec2 <- new (size $ extent arr1)
  loadP arr1 mvec2
  unsafeFreeze mvec2

{-# INLINE [4] computeP #-}
computeP :: (Load r1 sh e, Target r2 e, Source r2 e, Monad m) => Tensor r1 sh e -> m (Tensor r2 sh e)
computeP arr = now $ suspendedComputeP arr

{-# INLINE [4] computeS #-}
computeS :: (Load r1 sh e, Target r2 e) => Tensor r1 sh e -> Tensor r2 sh e
computeS arr1 = arr1 `seqTensor` unsafePerformIO $ do
  mvec2 <- new (size $ extent arr1)
  loadS arr1 mvec2
  unsafeFreeze mvec2

{-# INLINE [4] suspendedCopyP #-}
suspendedCopyP :: (Source r1 e, Load D sh e, Target r2 e) => Tensor r1 sh e -> Tensor r2 sh e
suspendedCopyP arr1 = suspendedComputeP $ delay arr1

{-# INLINE [4] copyP #-}
copyP :: (Source r1 e, Source r2 e, Load D sh e, Target r2 e, Monad m) => Tensor r1 sh e -> m (Tensor r2 sh e)
copyP arr = now $ suspendedCopyP arr

{-# INLINE [4] copyS #-}
copyS :: (Source r1 e, Load D sh e, Target r2 e) => Tensor r1 sh e -> Tensor r2 sh e
copyS arr1 = computeS $ delay arr1