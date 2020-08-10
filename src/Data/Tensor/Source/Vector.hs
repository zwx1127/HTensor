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

module Data.Tensor.Source.Vector where

import Control.Monad
import Data.Proxy (Proxy (..))
import Data.Tensor.Eval.Target
import Data.Tensor.Shape
import Data.Tensor.Source
import qualified Data.Vector as V
import qualified Data.Vector.Mutable as VM

data V

instance Source V e where
  data Tensor V sh e where
    TVector :: (Shape sh) => !(V.Vector e) -> Tensor V sh e

  {-# INLINE linearIndex #-}
  linearIndex (TVector vec) ix = vec V.! ix

  {-# INLINE unsafeLinearIndex #-}
  unsafeLinearIndex (TVector vec) ix = vec `V.unsafeIndex` ix

  {-# INLINE fromList #-}
  fromList :: forall sh. (Shape sh) => [e] -> Tensor V sh e
  fromList xs = TVector @sh (V.fromList xs)

  {-# INLINE generate #-}
  generate :: forall sh. (Shape sh) => (Index sh -> e) -> Tensor V sh e
  generate f =
    let f' = f . (fromIndex (Proxy @sh))
     in TVector @sh (V.generate (size (Proxy @sh)) f')

  {-# INLINE reshape #-}
  reshape :: forall sh1 sh2. (Shape sh1, Shape sh2) => Tensor V sh1 e -> Tensor V sh2 e
  reshape (TVector vec) = TVector @sh2 vec

  {-# INLINE seqTensor #-}
  seqTensor (TVector vec) x = vec `seq` x

  {-# INLINE tensorToString #-}
  tensorToString :: forall sh. (Shape sh, Show e) => Tensor V sh e -> String
  tensorToString (TVector vec) = "(" ++ (shapeToString (Proxy @sh)) ++ ")" ++ " " ++ (show vec)

instance Target V e where
  data MVec V e = MVector !(VM.IOVector e)

  {-# INLINE new #-}
  new n = liftM MVector (VM.new n)

  {-# INLINE unsafeWrite #-}
  unsafeWrite (MVector v) ix = VM.unsafeWrite v ix

  {-# INLINE unsafeRead #-}
  unsafeRead (MVector v) ix = VM.unsafeRead v ix

  {-# INLINE unsafeFreeze #-}
  unsafeFreeze :: forall sh. (Shape sh) => MVec V e -> IO (Tensor V sh e)
  unsafeFreeze (MVector mvec) = do
    vec <- V.unsafeFreeze mvec
    return $ TVector @sh vec

  {-# INLINE seqMVec #-}
  seqMVec vec x = vec `seq` x

zeroVTensor :: (Num e, Shape sh) => Tensor V sh e
zeroVTensor = repeatTensor 0

oneVTensor :: (Num e, Shape sh) => Tensor V sh e
oneVTensor = repeatTensor 1