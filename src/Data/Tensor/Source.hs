{-# LANGUAGE ConstraintKinds #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE EmptyDataDecls #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE PolyKinds #-}
{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE TypeOperators #-}

module Data.Tensor.Source where

import Data.Proxy (Proxy (..))
import Data.Tensor.Shape

class Source r e where
  data Tensor r sh e

  {-# INLINE extent #-}
  extent :: forall sh. (Shape sh) => Tensor r sh e -> Proxy sh
  extent _ = Proxy @sh

  (!), (!?), index, unsafeIndex :: forall sh. (Shape sh) => Tensor r sh e -> Index sh -> e
  {-# INLINE (!) #-}
  (!) = index
  {-# INLINE (!?) #-}
  (!?) = unsafeIndex
  {-# INLINE index #-}
  index ts ix = ts `linearIndex` (toIndex (Proxy @sh) ix)
  {-# INLINE unsafeIndex #-}
  unsafeIndex ts ix = ts `unsafeLinearIndex` (toIndex (Proxy @sh) ix)

  linearIndex, unsafeLinearIndex :: (Shape sh) => Tensor r sh e -> Int -> e
  {-# INLINE unsafeLinearIndex #-}
  unsafeLinearIndex = linearIndex

  fromList :: (Shape sh) => [e] -> Tensor r sh e

  generate :: (Shape sh) => (Index sh -> e) -> Tensor r sh e

  reshape :: (Shape sh1, Shape sh2) => Tensor r sh1 e -> Tensor r sh2 e

  seqTensor :: (Shape sh) => Tensor r sh e -> b -> b

  tensorToString :: (Shape sh, Show e) => Tensor r sh e -> String

-- instance (Source r e, Num e) => Num (Tensor r Z e) where
--   negate arr = generate (\ix -> negate $ arr !? ix)
--   (+) arr1 arr2 = generate (\ix -> (arr1 !? ix) + (arr2 !? ix))
--   (*) arr1 arr2 = generate (\ix -> (arr1 !? ix) * (arr2 !? ix))
--   fromInteger a = generate (\_ -> fromInteger a)
--   abs arr = generate (\ix -> abs $ arr !? ix)
--   signum arr = generate (\ix -> signum $ arr !? ix)

instance (Source r e, Shape sh, Show e) => Show (Tensor r sh e) where
  show = tensorToString

{-# INLINE repeatTensor #-}
repeatTensor :: (Source r e, Shape sh) => e -> Tensor r sh e
repeatTensor e = generate (\_ -> e)

{-# INLINE zeroTensor #-}
zeroTensor :: (Source r e, Num e, Shape sh) => Tensor r sh e
zeroTensor = repeatTensor 0

{-# INLINE oneTensor #-}
oneTensor :: (Source r e, Num e, Shape sh) => Tensor r sh e
oneTensor = repeatTensor 1
