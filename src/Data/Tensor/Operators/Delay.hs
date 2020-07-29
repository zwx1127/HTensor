{-# LANGUAGE PolyKinds #-}
{-# LANGUAGE RankNTypes #-}

module Data.Tensor.Operators.Delay where

import Data.Tensor.Shape
import Data.Tensor.Source
import Data.Tensor.Source.Delay

{-# INLINE [1] mapTensor #-}
mapTensor :: (Source r e, Shape sh) => (e -> e) -> Tensor r sh e -> Tensor D sh e
mapTensor f arr = generate $ gen f arr
  where
    gen gf a ix = gf (a !? ix)
