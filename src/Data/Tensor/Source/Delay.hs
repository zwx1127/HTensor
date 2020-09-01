{-# LANGUAGE GADTs #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE InstanceSigs #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeFamilies #-}

module Data.Tensor.Source.Delay where

import Data.Proxy (Proxy (..))
import Data.Tensor.Eval.Chunk
import Data.Tensor.Eval.Load
import Data.Tensor.Eval.Target
import Data.Tensor.Shape
import Data.Tensor.Source

data D

instance Source D e where
  data Tensor D sh e where
    TDelay :: (Shape sh) => (Index sh -> e) -> Tensor D sh e

  {-# INLINE linearIndex #-}
  linearIndex :: forall sh. (Shape sh) => Tensor D sh e -> Int -> e
  linearIndex (TDelay f) i = f (fromIndex (Proxy @sh) i)

  {-# INLINE fromList #-}
  fromList :: forall sh. (Shape sh) => [e] -> Tensor D sh e
  fromList xs = TDelay @sh (f xs)
    where
      f :: [e] -> Index sh -> e
      f xs' ix = xs' !! (toIndex (Proxy @sh) ix)

  {-# INLINE generate #-}
  generate :: forall sh. (Shape sh) => (Index sh -> e) -> Tensor D sh e
  generate f = TDelay @sh f

  {-# INLINE reshape #-}
  reshape :: forall sh1 sh2. (Shape sh1, Shape sh2) => Tensor D sh1 e -> Tensor D sh2 e
  reshape (TDelay f) = TDelay @sh2 (f . cast)
    where
      cast :: (Shape sh1, Shape sh2) => Index sh2 -> Index sh1
      cast = (fromIndex (Proxy @sh1)) . (toIndex (Proxy @sh2))

  {-# INLINE seqTensor #-}
  seqTensor (TDelay f) x = f `seq` x

  {-# INLINE tensorToString #-}
  tensorToString :: forall sh. (Shape sh, Show e) => Tensor D sh e -> String
  tensorToString _ = "(" ++ (shapeToString (Proxy @sh)) ++ ")" ++ " " ++ "Delayed"

instance forall sh e. (Shape sh) => Load D sh e where
  {-# INLINE [4] loadP #-}
  loadP (TDelay f) mvec = mvec `seqMVec` do
    fillChunkedP
      (size (Proxy @sh))
      (unsafeWrite mvec)
      (f . fromIndex (Proxy @sh))
  {-# INLINE [4] loadS #-}
  loadS (TDelay f) mvec = mvec `seqMVec` do
    fillLinearS
      (size (Proxy @sh))
      (unsafeWrite mvec)
      (f . fromIndex (Proxy @sh))

{-# INLINE delay #-}
delay :: forall r sh e. (Shape sh, Source r e) => Tensor r sh e -> Tensor D sh e
delay arr = TDelay @sh (unsafeIndex arr)

{-# INLINE zeroDTensor #-}
zeroDTensor :: (Num e, Shape sh) => Tensor D sh e
zeroDTensor = repeatTensor 0

{-# INLINE oneDTensor #-}
oneDTensor :: (Num e, Shape sh) => Tensor D sh e
oneDTensor = repeatTensor 1