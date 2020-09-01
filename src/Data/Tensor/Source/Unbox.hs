{-# LANGUAGE DataKinds #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE InstanceSigs #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE PolyKinds #-}
{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeFamilies #-}

module Data.Tensor.Source.Unbox where

import Control.Monad
import Data.Proxy (Proxy (..))
import Data.Tensor.Eval.Target
import Data.Tensor.Shape
import Data.Tensor.Source
import qualified Data.Vector.Unboxed as U
import qualified Data.Vector.Unboxed.Mutable as UM

data U

instance U.Unbox e => Source U e where
  data Tensor U sh e where
    TUnbox :: (Shape sh) => !(U.Vector e) -> Tensor U sh e

  {-# INLINE linearIndex #-}
  linearIndex (TUnbox vec) ix = vec U.! ix

  {-# INLINE unsafeLinearIndex #-}
  unsafeLinearIndex (TUnbox vec) ix = vec `U.unsafeIndex` ix

  {-# INLINE fromList #-}
  fromList :: forall sh. (Shape sh) => [e] -> Tensor U sh e
  fromList xs = TUnbox @sh (U.fromList xs)

  {-# INLINE generate #-}
  generate :: forall sh. (Shape sh) => (Index sh -> e) -> Tensor U sh e
  generate f =
    let f' = f . (fromIndex (Proxy @sh))
     in TUnbox @sh (U.generate (size (Proxy @sh)) f')

  {-# INLINE reshape #-}
  reshape :: forall sh1 sh2. (Shape sh1, Shape sh2) => Tensor U sh1 e -> Tensor U sh2 e
  reshape (TUnbox vec) = TUnbox @sh2 vec

  {-# INLINE seqTensor #-}
  seqTensor (TUnbox vec) x = vec `seq` x

  {-# INLINE tensorToString #-}
  tensorToString :: forall sh. (Shape sh, Show e) => Tensor U sh e -> String
  tensorToString (TUnbox vec) = "(" ++ (shapeToString (Proxy @sh)) ++ ")" ++ " " ++ (show vec)

instance (U.Unbox e) => Target U e where
  data MVec U e = UVector !(UM.IOVector e)

  {-# INLINE new #-}
  new n = liftM UVector (UM.new n)

  {-# INLINE unsafeWrite #-}
  unsafeWrite (UVector v) ix = UM.unsafeWrite v ix

  {-# INLINE unsafeRead #-}
  unsafeRead (UVector v) ix = UM.unsafeRead v ix

  {-# INLINE unsafeFreeze #-}
  unsafeFreeze :: forall sh. (Shape sh) => MVec U e -> IO (Tensor U sh e)
  unsafeFreeze (UVector mvec) = do
    vec <- U.unsafeFreeze mvec
    return $ TUnbox @sh vec

  {-# INLINE seqMVec #-}
  seqMVec vec x = vec `seq` x

{-# INLINE zeroUTensor #-}
zeroUTensor :: (U.Unbox e, Num e, Shape sh) => Tensor U sh e
zeroUTensor = repeatTensor 0

{-# INLINE oneUTensor #-}
oneUTensor :: (U.Unbox e, Num e, Shape sh) => Tensor U sh e
oneUTensor = repeatTensor 1