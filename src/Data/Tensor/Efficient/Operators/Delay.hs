{-# LANGUAGE PolyKinds #-}
{-# LANGUAGE RankNTypes #-}

module Data.Tensor.Efficient.Operators.Delay where

import Data.Tensor.Efficient.Shape
import Data.Tensor.Efficient.Source
import Data.Tensor.Efficient.Source.Delay

{-# INLINE [1] mapTensor #-}
mapTensor :: forall (r :: *) e sh. (Source r e, Shape sh) => (e -> e) -> Tensor r sh e -> Tensor D sh e
mapTensor f arr = generate $ gen f arr
  where
    gen gf a ix = gf (a !? ix)
