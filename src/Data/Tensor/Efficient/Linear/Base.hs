{-# LANGUAGE DataKinds #-}
{-# LANGUAGE PolyKinds #-}
{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeOperators #-}

module Data.Tensor.Efficient.Linear.Base where

import Data.Tensor.Efficient.Shape
import Data.Tensor.Efficient.Source
import GHC.TypeLits (KnownNat)

-- type Scalar r e = Tensor r Z e

type Matrix r m n e = Tensor r (m >< n >< Z) e

type Vector r m e = Tensor r (m >< Z) e

type RVector r m e = Matrix r 1 m e

type CVector r m e = Matrix r m 1 e

rv :: (Source r e, KnownNat m, Num e) => Vector r m e -> RVector r m e
rv = reshape

cv :: (Source r e, KnownNat m, Num e) => Vector r m e -> CVector r m e
cv = reshape

rv2cv :: (Source r e, KnownNat m, Num e) => RVector r m e -> CVector r m e
rv2cv = reshape

cv2rv :: (Source r e, KnownNat m, Num e) => CVector r m e -> RVector r m e
cv2rv = reshape
