{-# LANGUAGE MagicHash #-}
{-# LANGUAGE BangPatterns #-}

module Data.Tensor.Eval.Chunk where

import Data.Tensor.Eval.Gang
import GHC.Exts

fillLinearS :: Int -> (Int -> a -> IO ()) -> (Int -> a) -> IO ()
fillLinearS !(I# len) write f = fill 0#
  where
    fill !ix
      | 1# <- ix >=# len = return ()
      | otherwise = do
        write (I# ix) (f (I# ix))
        fill (ix +# 1#)

fillChunkedP :: Int -> (Int -> a -> IO ()) -> (Int -> a) -> IO ()
fillChunkedP !(I# len) write f = gangIO theGang $ \(I# thread) ->
  let !start = splitIx thread
      !end = splitIx (thread +# 1#)
   in fill start end
  where
    !(I# threads) = count theGang
    !chunkLen = len `quotInt#` threads
    !chunkLeftover = len `remInt#` threads
    {-# INLINE splitIx #-}
    splitIx thread
      | 1# <- thread <# chunkLeftover = thread *# (chunkLen +# 1#)
      | otherwise = thread *# chunkLen +# chunkLeftover
    {-# INLINE fill #-}
    fill !ix !end
      | 1# <- ix >=# end = return ()
      | otherwise = do
        write (I# ix) (f (I# ix))
        fill (ix +# 1#) end
