import React from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { Camera, Zap, Target, User, Layers, Brain, MessageSquare, Volume2, CheckCircle, Circle } from 'lucide-react'

const MODULE_STAGES = {
  module1: [
    { id: 'capture', label: 'Webcam Capture', icon: Camera, duration: 800 },
    { id: 'motion', label: 'Motion Filter', icon: Zap, duration: 600 },
    { id: 'roi', label: 'ROI Extraction', icon: Target, duration: 700 },
    { id: 'pose', label: 'Pose Detection', icon: User, duration: 900 },
  ],
  module2: [
    { id: 'rgb', label: 'RGB Stream (CNN)', icon: Layers, duration: 1000 },
    { id: 'pose_feat', label: 'Pose Stream (MLP)', icon: User, duration: 800 },
    { id: 'fusion', label: 'Attention Fusion', icon: Zap, duration: 700 },
    { id: 'features', label: 'Feature Vector', icon: CheckCircle, duration: 600 },
  ],
  module3: [
    { id: 'temporal', label: 'BiLSTM Processing', icon: Brain, duration: 1200 },
    { id: 'ctc', label: 'CTC Alignment', icon: Layers, duration: 900 },
    { id: 'decode', label: 'Sign Decoding', icon: MessageSquare, duration: 800 },
    { id: 'gloss', label: 'Gloss Output', icon: CheckCircle, duration: 600 },
  ],
  module4: [
    { id: 'translate', label: 'Gloss-to-Text', icon: MessageSquare, duration: 900 },
    { id: 'grammar', label: 'Grammar Correction', icon: Brain, duration: 800 },
    { id: 'tts', label: 'Text-to-Speech', icon: Volume2, duration: 1000 },
    { id: 'output', label: 'Audio Output', icon: CheckCircle, duration: 600 },
  ],
}

function StageIndicator({ stage, isActive, isComplete }) {
  const Icon = stage.icon
  
  return (
    <motion.div
      initial={{ opacity: 0, x: -20 }}
      animate={{ opacity: 1, x: 0 }}
      className={`flex items-center gap-3 rounded-xl px-4 py-3 ${
        isActive
          ? 'bg-cyan-500/30 border-2 border-cyan-300'
          : isComplete
          ? 'bg-emerald-500/20 border border-emerald-400/50'
          : 'bg-slate-700/30 border border-slate-600/30'
      }`}
    >
      <div className="relative">
        <Icon
          size={24}
          className={
            isActive
              ? 'text-cyan-300'
              : isComplete
              ? 'text-emerald-300'
              : 'text-slate-400'
          }
        />
        {isActive && (
          <motion.div
            animate={{ scale: [1, 1.3, 1], opacity: [0.6, 0, 0.6] }}
            transition={{ duration: 1.5, repeat: Infinity }}
            className="absolute inset-0 rounded-full bg-cyan-400"
          />
        )}
      </div>
      
      <div className="flex-1">
        <div className={`font-semibold ${isActive ? 'text-cyan-100' : isComplete ? 'text-emerald-100' : 'text-slate-300'}`}>
          {stage.label}
        </div>
        {isActive && (
          <div className="mt-1 h-1 overflow-hidden rounded-full bg-black/30">
            <motion.div
              initial={{ width: '0%' }}
              animate={{ width: '100%' }}
              transition={{ duration: stage.duration / 1000, ease: 'linear' }}
              className="h-full bg-gradient-to-r from-cyan-400 to-blue-400"
            />
          </div>
        )}
      </div>
      
      <div>
        {isComplete ? (
          <CheckCircle size={20} className="text-emerald-400" />
        ) : isActive ? (
          <motion.div
            animate={{ rotate: 360 }}
            transition={{ duration: 1, repeat: Infinity, ease: 'linear' }}
          >
            <Circle size={20} className="text-cyan-400" />
          </motion.div>
        ) : (
          <Circle size={20} className="text-slate-500" />
        )}
      </div>
    </motion.div>
  )
}

function VisualizationArea({ module, activeStageId }) {
  const getVisualization = () => {
    switch (module) {
      case 'module1':
        return (
          <div className="relative h-64 overflow-hidden rounded-2xl border-2 border-cyan-300/50 bg-gradient-to-br from-slate-900 via-slate-800 to-cyan-900/30">
            <motion.div
              animate={{
                scale: [1, 1.05, 1],
                opacity: [0.5, 0.8, 0.5],
              }}
              transition={{ duration: 2, repeat: Infinity }}
              className="absolute inset-0 bg-gradient-to-br from-cyan-500/20 to-blue-500/20"
            />
            
            {/* Fake video feed grid */}
            <div className="absolute inset-4 grid grid-cols-8 gap-1">
              {Array.from({ length: 64 }).map((_, i) => (
                <motion.div
                  key={i}
                  animate={{
                    opacity: [0.2, 0.6, 0.2],
                    backgroundColor: activeStageId === 'pose' 
                      ? ['#0f172a', '#22d3ee', '#0f172a']
                      : ['#0f172a', '#475569', '#0f172a']
                  }}
                  transition={{
                    duration: 1.5,
                    delay: i * 0.02,
                    repeat: Infinity,
                  }}
                  className="aspect-square rounded-sm"
                />
              ))}
            </div>
            
            {activeStageId === 'pose' && (
              <div className="absolute inset-0 flex items-center justify-center">
                <motion.div
                  animate={{ scale: [0.8, 1.2, 0.8], opacity: [0.3, 0.7, 0.3] }}
                  transition={{ duration: 2, repeat: Infinity }}
                  className="text-6xl"
                >
                  🤚
                </motion.div>
              </div>
            )}
            
            <div className="absolute bottom-4 left-4 rounded-lg bg-black/60 px-3 py-1 font-mono text-sm text-cyan-300">
              {activeStageId === 'capture' && '640×480 @ 20 FPS'}
              {activeStageId === 'motion' && 'Motion Score: 12.5'}
              {activeStageId === 'roi' && 'ROI: 224×224'}
              {activeStageId === 'pose' && '75 Keypoints Detected'}
            </div>
          </div>
        )
        
      case 'module2':
        return (
          <div className="relative h-64 overflow-hidden rounded-2xl border-2 border-purple-300/50 bg-gradient-to-br from-slate-900 via-purple-900/20 to-slate-800">
            <div className="absolute inset-0 flex items-center justify-center gap-8">
              {/* RGB Stream */}
              <motion.div
                animate={{
                  y: [0, -10, 0],
                  boxShadow: activeStageId === 'rgb' 
                    ? ['0 0 0px rgba(168,85,247,0)', '0 0 30px rgba(168,85,247,0.8)', '0 0 0px rgba(168,85,247,0)']
                    : '0 0 0px rgba(168,85,247,0)',
                }}
                transition={{ duration: 2, repeat: Infinity }}
                className="relative"
              >
                <div className="h-32 w-32 rounded-2xl bg-gradient-to-br from-purple-500 to-pink-500 p-1">
                  <div className="h-full w-full rounded-xl bg-slate-900 flex items-center justify-center">
                    <Layers size={48} className="text-purple-300" />
                  </div>
                </div>
                <div className="mt-2 text-center text-sm font-bold text-purple-300">RGB Stream</div>
              </motion.div>
              
              {/* Fusion */}
              {activeStageId === 'fusion' && (
                <motion.div
                  initial={{ scale: 0, opacity: 0 }}
                  animate={{ scale: 1, opacity: 1 }}
                  className="absolute"
                >
                  <Zap size={64} className="text-yellow-400" />
                </motion.div>
              )}
              
              {/* Pose Stream */}
              <motion.div
                animate={{
                  y: [0, -10, 0],
                  boxShadow: activeStageId === 'pose_feat'
                    ? ['0 0 0px rgba(34,211,238,0)', '0 0 30px rgba(34,211,238,0.8)', '0 0 0px rgba(34,211,238,0)']
                    : '0 0 0px rgba(34,211,238,0)',
                }}
                transition={{ duration: 2, repeat: Infinity, delay: 0.5 }}
                className="relative"
              >
                <div className="h-32 w-32 rounded-2xl bg-gradient-to-br from-cyan-500 to-blue-500 p-1">
                  <div className="h-full w-full rounded-xl bg-slate-900 flex items-center justify-center">
                    <User size={48} className="text-cyan-300" />
                  </div>
                </div>
                <div className="mt-2 text-center text-sm font-bold text-cyan-300">Pose Stream</div>
              </motion.div>
            </div>
            
            {activeStageId === 'features' && (
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                className="absolute bottom-4 left-1/2 -translate-x-1/2 rounded-xl bg-emerald-500/20 border border-emerald-400 px-6 py-3"
              >
                <div className="text-center font-mono text-sm font-bold text-emerald-300">
                  Features: 768-dim vector
                </div>
              </motion.div>
            )}
          </div>
        )
        
      case 'module3':
        return (
          <div className="relative h-64 overflow-hidden rounded-2xl border-2 border-amber-300/50 bg-gradient-to-br from-slate-900 via-amber-900/20 to-slate-800">
            <div className="absolute inset-0 flex items-center justify-center">
              {/* Neural network visualization */}
              <div className="relative w-full max-w-md">
                {/* BiLSTM layers */}
                {[0, 1, 2, 3, 4].map((layer) => (
                  <motion.div
                    key={layer}
                    initial={{ opacity: 0, x: -50 }}
                    animate={{
                      opacity: activeStageId === 'temporal' ? [0.3, 1, 0.3] : 0.3,
                      x: 0,
                    }}
                    transition={{
                      duration: 1.5,
                      delay: layer * 0.2,
                      repeat: Infinity,
                    }}
                    className="mb-3 flex items-center justify-center gap-2"
                  >
                    {Array.from({ length: 8 }).map((_, i) => (
                      <div
                        key={i}
                        className="h-8 w-8 rounded-full bg-gradient-to-br from-amber-400 to-orange-500"
                      />
                    ))}
                  </motion.div>
                ))}
              </div>
              
              {activeStageId === 'gloss' && (
                <motion.div
                  initial={{ opacity: 0, scale: 0.8 }}
                  animate={{ opacity: 1, scale: 1 }}
                  className="absolute inset-0 flex items-center justify-center bg-black/70"
                >
                  <div className="text-center">
                    <div className="text-5xl font-bold text-amber-300">
                      HELLO
                    </div>
                    <div className="mt-2 text-lg text-slate-300">
                      Sign Detected
                    </div>
                  </div>
                </motion.div>
              )}
            </div>
          </div>
        )
        
      case 'module4':
        return (
          <div className="relative h-64 overflow-hidden rounded-2xl border-2 border-emerald-300/50 bg-gradient-to-br from-slate-900 via-emerald-900/20 to-slate-800">
            <div className="absolute inset-0 flex flex-col items-center justify-center gap-4 p-8">
              {activeStageId === 'translate' && (
                <motion.div
                  initial={{ opacity: 0, y: -20 }}
                  animate={{ opacity: 1, y: 0 }}
                  className="text-3xl font-bold text-amber-300"
                >
                  ME GO SCHOOL
                </motion.div>
              )}
              
              {(activeStageId === 'grammar' || activeStageId === 'tts' || activeStageId === 'output') && (
                <>
                  <motion.div
                    initial={{ opacity: 0.3 }}
                    animate={{ opacity: 1 }}
                    className="text-xl text-slate-400 line-through"
                  >
                    ME GO SCHOOL
                  </motion.div>
                  <motion.div
                    initial={{ opacity: 0, scale: 0.9 }}
                    animate={{ opacity: 1, scale: 1 }}
                    className="text-4xl font-bold text-emerald-300"
                  >
                    "I am going to school."
                  </motion.div>
                </>
              )}
              
              {activeStageId === 'output' && (
                <motion.div
                  animate={{ scale: [1, 1.2, 1] }}
                  transition={{ duration: 1, repeat: Infinity }}
                  className="mt-4"
                >
                  <Volume2 size={64} className="text-emerald-400" />
                </motion.div>
              )}
            </div>
          </div>
        )
        
      default:
        return null
    }
  }
  
  return (
    <AnimatePresence mode="wait">
      <motion.div
        key={module + activeStageId}
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        exit={{ opacity: 0 }}
      >
        {getVisualization()}
      </motion.div>
    </AnimatePresence>
  )
}

export default function ModuleSimulation({ selectedModule, isRunning }) {
  const stages = MODULE_STAGES[selectedModule] || []
  const [activeStageIndex, setActiveStageIndex] = React.useState(0)
  const [completedStages, setCompletedStages] = React.useState(new Set())
  
  React.useEffect(() => {
    if (!isRunning) {
      setActiveStageIndex(0)
      setCompletedStages(new Set())
      return
    }
    
    let timeout
    const currentStage = stages[activeStageIndex]
    
    if (currentStage) {
      timeout = setTimeout(() => {
        setCompletedStages((prev) => new Set([...prev, currentStage.id]))
        
        if (activeStageIndex < stages.length - 1) {
          setActiveStageIndex((prev) => prev + 1)
        } else {
          // Loop back
          setTimeout(() => {
            setActiveStageIndex(0)
            setCompletedStages(new Set())
          }, 1000)
        }
      }, currentStage.duration)
    }
    
    return () => clearTimeout(timeout)
  }, [isRunning, activeStageIndex, selectedModule])
  
  const activeStage = stages[activeStageIndex]
  
  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className="glass-panel rounded-3xl p-6"
    >
      <div className="mb-4 flex items-center justify-between">
        <h3 className="text-2xl font-bold text-slate-100">
          {selectedModule === 'module1' && 'Module 1: Video Preprocessing'}
          {selectedModule === 'module2' && 'Module 2: Feature Extraction'}
          {selectedModule === 'module3' && 'Module 3: Temporal Recognition'}
          {selectedModule === 'module4' && 'Module 4: Translation & Output'}
        </h3>
        <div className={`rounded-full px-4 py-1 text-sm font-bold ${isRunning ? 'bg-emerald-500/30 text-emerald-300' : 'bg-slate-700/30 text-slate-400'}`}>
          {isRunning ? 'RUNNING' : 'IDLE'}
        </div>
      </div>
      
      <VisualizationArea
        module={selectedModule}
        activeStageId={activeStage?.id}
      />
      
      <div className="mt-6 space-y-3">
        {stages.map((stage, idx) => (
          <StageIndicator
            key={stage.id}
            stage={stage}
            isActive={isRunning && idx === activeStageIndex}
            isComplete={completedStages.has(stage.id)}
          />
        ))}
      </div>
    </motion.div>
  )
}
