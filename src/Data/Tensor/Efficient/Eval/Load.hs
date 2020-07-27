{-# LANGUAGE MultiParamTypeClasses #-}

module Data.Tensor.Efficient.Eval.Load where

import           Data.Tensor.Efficient.Source
import           Data.Tensor.Efficient.Shape
import           Data.Tensor.Efficient.Eval.Target

class (Source r1 e, Shape sh) => Load r1 sh e where

    loadS :: (Target r2 e) => Tensor r1 sh e -> MVec r2 e -> IO ()

    loadP :: (Target r2 e) => Tensor r1 sh e -> MVec r2 e -> IO ()
