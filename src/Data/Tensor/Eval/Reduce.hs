{-# LANGUAGE BangPatterns #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE MagicHash #-}

module Data.Tensor.Eval.Reduce where

import Data.Tensor.Eval.Gang
import qualified Data.Vector.Unboxed as V
import qualified Data.Vector.Unboxed.Mutable as M
import GHC.Base (divInt, quotInt)
import GHC.Exts

{-# INLINE [0] reduceAny #-}
reduceAny :: (Int# -> a) -> (a -> a -> a) -> a -> Int# -> Int# -> a
reduceAny f c !r !start !end = iter start r
  where
    {-# INLINE iter #-}
    iter !i !z
      | 1# <- i >=# end = z
      | otherwise = iter (i +# 1#) (z `c` f i)

{-# INLINE [0] reduce #-}
reduce :: (Int -> a) -> (a -> a -> a) -> a -> Int -> Int -> a
reduce f c !r (I# start) (I# end) = reduceAny (\i -> f (I# i)) c r start end

{-# INLINE [1] foldS #-}
foldS :: (V.Unbox) a => M.IOVector a -> (Int# -> a) -> (a -> a -> a) -> a -> Int# -> IO ()
foldS !vec get c !r !n = iter 0# 0#
  where
    !(I# end) = M.length vec
    {-# INLINE iter #-}
    iter !sh !sz
      | 1# <- sh >=# end = return ()
      | otherwise = do
        let !next = sz +# n
        M.unsafeWrite vec (I# sh) (reduceAny get c r sz next)
        iter (sh +# 1#) next

{-# INLINE [1] foldP #-}
foldP :: (V.Unbox a) => M.IOVector a -> (Int -> a) -> (a -> a -> a) -> a -> Int -> IO ()
foldP vec f c !r (I# n) = gangIO theGang $ \(I# tid) -> fill (split tid) (split (tid +# 1#))
  where
    !(I# threads) = count theGang
    !(I# len) = M.length vec
    !step = (len +# threads -# 1#) `quotInt#` threads
    {-# INLINE split #-}
    split !ix =
      let !ix' = ix *# step
       in case len <# ix' of
            0# -> ix'
            _ -> len
    {-# INLINE fill #-}
    fill !start !end = iter start (start *# n)
      where
        {-# INLINE iter #-}
        iter !sh !sz
          | 1# <- sh >=# end = return ()
          | otherwise = do
            let !next = sz +# n
            M.unsafeWrite vec (I# sh) (reduce f c r (I# sz) (I# next))
            iter (sh +# 1#) next

{-# INLINE [1] foldAllS #-}
foldAllS :: (Int# -> a) -> (a -> a -> a) -> a -> Int# -> a
foldAllS f c !r !len = reduceAny (\i -> f i) c r 0# len

{-# INLINE [1] foldAllP #-}
foldAllP :: (V.Unbox a) => (Int -> a) -> (a -> a -> a) -> a -> Int -> IO a
foldAllP f c !r !len
  | len == 0 = return r
  | otherwise = do
    mvec <- M.unsafeNew chunks
    gangIO theGang $ \tid -> fill mvec tid (split tid) (split (tid + 1))
    vec <- V.unsafeFreeze mvec
    return $! V.foldl' c r vec
  where
    !threads = count theGang
    !step = (len + threads - 1) `quotInt` threads
    chunks = ((len + step - 1) `divInt` step) `min` threads
    {-# INLINE split #-}
    split !ix = len `min` (ix * step)
    {-# INLINE fill #-}
    fill !mvec !tid !start !end
      | start >= end = return ()
      | otherwise = M.unsafeWrite mvec tid (reduce f c (f start) (start + 1) end)