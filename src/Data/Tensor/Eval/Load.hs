{-# LANGUAGE MultiParamTypeClasses #-}

module Data.Tensor.Eval.Load where

import           Data.Tensor.Source
import           Data.Tensor.Shape
import           Data.Tensor.Eval.Target

class (Source r1 e, Shape sh) => Load r1 sh e where

    loadS :: (Target r2 e) => Tensor r1 sh e -> MVec r2 e -> IO ()

    loadP :: (Target r2 e) => Tensor r1 sh e -> MVec r2 e -> IO ()
