module Data.Tensor.Efficient.Eval.Gang where

import Control.Concurrent.MVar
import Control.Monad
import GHC.Conc (forkOn)
import GHC.Conc (numCapabilities)
import GHC.IO

theGang :: Gang
{-# NOINLINE theGang #-}
theGang = unsafePerformIO $ do
  let caps = numCapabilities
  forkGang caps

data Request = Action (Int -> IO ()) | Shutdown

data Gang = Gang
  { count :: !Int,
    requests :: [MVar Request],
    results :: [MVar ()],
    isBusy :: MVar Bool
  }

instance Show Gang where
  show Gang {count = n} = "<" ++ show n ++ " threads>"

gangWorker :: Int -> MVar Request -> MVar () -> IO ()
gangWorker threadId request result = do
  req <- takeMVar request
  case req of
    Action f -> do
      f threadId
      putMVar result ()
      gangWorker threadId request result
    Shutdown -> putMVar result ()

finaliseWorker :: MVar Request -> MVar () -> IO ()
finaliseWorker request result = do
  putMVar request Shutdown
  takeMVar result
  return ()

forkGang :: Int -> IO Gang
forkGang n
  | n > 0 = do
    requests' <- sequence $ replicate n $ newEmptyMVar
    results' <- sequence $ replicate n $ newEmptyMVar
    zipWithM_
      (\request result -> mkWeakMVar request (finaliseWorker request result))
      requests'
      results'
    zipWithM_ forkOn [0 ..] $ zipWith3 gangWorker [0 .. n - 1] requests' results'
    busy <- newMVar False
    return $ Gang n requests' results' busy
  | otherwise = error "fork gang number mast > 0"

{-# NOINLINE gangIO #-}
gangIO :: Gang -> (Int -> IO ()) -> IO ()
gangIO gang@Gang {isBusy = isBusy'} action = do
  b <- swapMVar isBusy' True
  if b
    then do
      seqIO gang action
    else do
      parIO gang action
      _ <- swapMVar isBusy' False
      return ()

seqIO :: Gang -> (Int -> IO ()) -> IO ()
seqIO Gang {count = n} action = do
  mapM_ action [0 .. n - 1]

parIO :: Gang -> (Int -> IO ()) -> IO ()
parIO Gang {requests = requests', results = results'} action = do
  mapM_ (\v -> putMVar v (Action action)) requests'
  mapM_ takeMVar results'
