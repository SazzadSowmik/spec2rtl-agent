import React, { useState } from 'react';

const RISCV32IKnowledgeGraph = () => {
  const [selectedNode, setSelectedNode] = useState(null);
  const [hoveredNode, setHoveredNode] = useState(null);

  const modules = [
    {
      id: 'RegisterFile',
      order: 1,
      level: 0,
      category: 'core',
      description: 'Implements a register file containing 32 general-purpose registers and a program counter for the RV32I architecture.',
      inputs: ['read_addr1 (5-bit)', 'read_addr2 (5-bit)', 'write_addr (5-bit)', 'write_data (32-bit)', 'write_enable (1-bit)'],
      outputs: ['read_data1 (32-bit)', 'read_data2 (32-bit)', 'pc (32-bit)'],
      references: ['Section 2.1', 'Figure 2.1'],
      dependencies: [],
      keyFeatures: ['32 registers × 32-bit', 'x0 hardwired to zero', '2 read ports, 1 write port']
    },
    {
      id: 'InstructionFetch',
      order: 2,
      level: 1,
      category: 'fetch',
      description: 'Fetches instructions from memory based on the program counter and handles instruction alignment.',
      inputs: ['pc (32-bit)', 'memory (byte array)', 'endian (1-bit)'],
      outputs: ['instruction (32-bit)', 'pc_next (32-bit)'],
      references: ['Section 2.6', 'Figure 2.1'],
      dependencies: ['RegisterFile'],
      keyFeatures: ['32-bit instruction fetch', 'Alignment checking', 'Endianness handling']
    },
    {
      id: 'InstructionDecode',
      order: 3,
      level: 2,
      category: 'decode',
      description: 'Decodes fetched instructions into control signals and identifies the instruction type (R, I, S, U, B, J).',
      inputs: ['instruction (32-bit)'],
      outputs: ['opcode (7-bit)', 'rd (5-bit)', 'rs1 (5-bit)', 'rs2 (5-bit)', 'imm (32-bit)', 'instruction_type'],
      references: ['Section 2.2', 'Figure 2.2'],
      dependencies: ['InstructionFetch'],
      keyFeatures: ['6 instruction formats', 'Field extraction', 'Type identification']
    },
    {
      id: 'ImmediateGenerator',
      order: 4,
      level: 3,
      category: 'decode',
      description: 'Generates immediate values from instruction formats, handling sign-extension and encoding variants.',
      inputs: ['instruction (32-bit)', 'instruction_type'],
      outputs: ['immediate (32-bit signed)'],
      references: ['Section 2.3', 'Figure 2.3', 'Figure 2.4'],
      dependencies: ['InstructionDecode'],
      keyFeatures: ['I/S/B/U/J immediate types', 'Sign extension', 'Bit rearrangement']
    },
    {
      id: 'ALU',
      order: 5,
      level: 3,
      category: 'execute',
      description: 'Performs arithmetic and logic operations based on control signals and input operands.',
      inputs: ['operand1 (32-bit)', 'operand2 (32-bit)', 'alu_control (4-bit)'],
      outputs: ['result (32-bit)', 'zero_flag (1-bit)', 'overflow_flag (1-bit)'],
      references: ['Section 2.4', 'Figure 2.3'],
      dependencies: ['ImmediateGenerator', 'InstructionDecode'],
      keyFeatures: ['ADD/SUB/AND/OR/XOR', 'SLL/SRL/SRA shifts', 'Comparison operations']
    },
    {
      id: 'ControlUnit',
      order: 6,
      level: 3,
      category: 'control',
      description: 'Generates control signals for the ALU, memory access, and register file based on the decoded instruction.',
      inputs: ['opcode (7-bit)', 'funct3 (3-bit)', 'funct7 (7-bit)', 'instruction_type'],
      outputs: ['alu_control (4-bit)', 'mem_read (1-bit)', 'mem_write (1-bit)', 'reg_write (1-bit)', 'branch (1-bit)', 'jump (1-bit)', 'link (1-bit)'],
      references: ['Section 2.5', 'Table 2.1'],
      dependencies: ['InstructionDecode'],
      keyFeatures: ['ALU operation select', 'Memory control', 'Branch/Jump control']
    },
    {
      id: 'MemoryAccess',
      order: 7,
      level: 4,
      category: 'memory',
      description: 'Handles load and store operations, ensuring proper alignment and addressing based on instruction type.',
      inputs: ['address (32-bit)', 'data (32-bit)', 'mem_read (1-bit)', 'mem_write (1-bit)', 'endian (1-bit)', 'data_type'],
      outputs: ['loaded_data (32-bit)', 'exception (1-bit)'],
      references: ['Section 2.6'],
      dependencies: ['ALU', 'ControlUnit'],
      keyFeatures: ['LB/LH/LW/LBU/LHU', 'SB/SH/SW', 'Alignment handling']
    },
    {
      id: 'ControlTransfer',
      order: 8,
      level: 4,
      category: 'control',
      description: 'Manages control transfer instructions, including jumps and branches, and updates the program counter accordingly.',
      inputs: ['instruction (32-bit)', 'current_pc (32-bit)', 'branch_taken (1-bit)', 'rs1_value (32-bit)', 'rd (5-bit)'],
      outputs: ['next_pc (32-bit)', 'exception (1-bit)'],
      references: ['Section 2.5', 'Table 2.1'],
      dependencies: ['ControlUnit', 'ALU'],
      keyFeatures: ['JAL/JALR jumps', 'BEQ/BNE/BLT/BGE branches', 'Return-address stack']
    },
    {
      id: 'HintHandler',
      order: 9,
      level: 3,
      category: 'optional',
      description: 'Processes HINT instructions to provide performance optimization suggestions without altering program behavior.',
      inputs: ['instruction (32-bit)', 'current_pc (32-bit)'],
      outputs: ['next_pc (32-bit)', 'performance_metrics'],
      references: ['Section 2.9', 'Table 2.3'],
      dependencies: ['InstructionDecode'],
      keyFeatures: ['rd=x0 encoding', 'Performance hints', 'No architectural change']
    },
    {
      id: 'VLIWSupport',
      order: 10,
      level: 3,
      category: 'extension',
      description: 'Supports Very Long Instruction Word (VLIW) encodings for executing multiple operations in a single instruction.',
      inputs: ['vliw_instruction (64-bit)', 'current_pc (32-bit)'],
      outputs: ['next_pc (32-bit)', 'operation_results (array)'],
      references: ['Section 26.5'],
      dependencies: ['InstructionDecode'],
      keyFeatures: ['64-bit instruction', 'Parallel execution', 'Multiple operations']
    },
    {
      id: 'ISAExtensions',
      order: 11,
      level: 3,
      category: 'extension',
      description: 'Manages machine-level instruction-set extensions to enhance the capabilities of the RISC-V architecture.',
      inputs: ['extension_id (5-bit)', 'instruction (32-bit)', 'current_pc (32-bit)'],
      outputs: ['next_pc (32-bit)', 'result (32-bit)', 'exception (1-bit)'],
      references: ['Section 27.9', 'Section 27.10', 'Section 27.11'],
      dependencies: ['InstructionDecode'],
      keyFeatures: ['M/A/F/D extensions', 'Custom extensions', 'Backward compatible']
    }
  ];

  const categoryColors = {
    core: { bg: '#3b82f6', border: '#1d4ed8', text: 'Core' },
    fetch: { bg: '#8b5cf6', border: '#6d28d9', text: 'Fetch' },
    decode: { bg: '#f59e0b', border: '#d97706', text: 'Decode' },
    execute: { bg: '#10b981', border: '#059669', text: 'Execute' },
    control: { bg: '#ef4444', border: '#dc2626', text: 'Control' },
    memory: { bg: '#06b6d4', border: '#0891b2', text: 'Memory' },
    optional: { bg: '#6b7280', border: '#4b5563', text: 'Optional' },
    extension: { bg: '#ec4899', border: '#db2777', text: 'Extension' }
  };

  const getNodePositions = () => {
    const positions = {};
    const levelGroups = {};
    
    modules.forEach(m => {
      if (!levelGroups[m.level]) levelGroups[m.level] = [];
      levelGroups[m.level].push(m.id);
    });

    const svgWidth = 880;
    const startY = 50;
    const levelHeight = 95;

    Object.keys(levelGroups).forEach(level => {
      const nodes = levelGroups[level];
      const levelY = startY + parseInt(level) * levelHeight;
      const spacing = svgWidth / (nodes.length + 1);
      
      nodes.forEach((nodeId, idx) => {
        positions[nodeId] = {
          x: spacing * (idx + 1),
          y: levelY
        };
      });
    });

    return positions;
  };

  const positions = getNodePositions();

  const getEdges = () => {
    const edges = [];
    modules.forEach(module => {
      module.dependencies.forEach(dep => {
        edges.push({
          from: dep,
          to: module.id,
          fromPos: positions[dep],
          toPos: positions[module.id]
        });
      });
    });
    return edges;
  };

  const edges = getEdges();

  const getPath = (from, to) => {
    const midY = (from.y + to.y) / 2;
    return `M ${from.x} ${from.y + 20} 
            Q ${from.x} ${midY}, ${(from.x + to.x) / 2} ${midY}
            Q ${to.x} ${midY}, ${to.x} ${to.y - 20}`;
  };

  return (
    <div className="w-full bg-slate-900 p-3 rounded-lg">
      <div className="text-center mb-3">
        <h1 className="text-lg font-bold text-white mb-1">RISC-V 32I Implementation Knowledge Graph</h1>
        <p className="text-slate-400 text-xs">Click on any module to view detailed specifications</p>
      </div>

      <div className="flex flex-wrap justify-center gap-2 mb-3">
        {Object.entries(categoryColors).map(([key, val]) => (
          <div key={key} className="flex items-center gap-1">
            <div className="w-3 h-3 rounded" style={{ backgroundColor: val.bg }}></div>
            <span className="text-slate-300 text-xs">{val.text}</span>
          </div>
        ))}
      </div>

      <div className="bg-slate-800 rounded-lg p-2 mb-3">
        <svg viewBox="0 0 880 520" className="w-full h-auto">
          <defs>
            <marker id="arrowhead" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto">
              <polygon points="0 0, 10 3.5, 0 7" fill="#64748b" />
            </marker>
            <filter id="glow">
              <feGaussianBlur stdDeviation="2" result="coloredBlur"/>
              <feMerge>
                <feMergeNode in="coloredBlur"/>
                <feMergeNode in="SourceGraphic"/>
              </feMerge>
            </filter>
          </defs>

          {[0, 1, 2, 3, 4].map(level => (
            <text key={level} x="12" y={50 + level * 95 + 4} fill="#64748b" fontSize="10" fontWeight="500">
              L{level}
            </text>
          ))}

          {edges.map((edge, idx) => (
            <path
              key={idx}
              d={getPath(edge.fromPos, edge.toPos)}
              fill="none"
              stroke={hoveredNode === edge.to || hoveredNode === edge.from ? '#94a3b8' : '#475569'}
              strokeWidth={hoveredNode === edge.to || hoveredNode === edge.from ? 2 : 1.5}
              markerEnd="url(#arrowhead)"
              opacity={hoveredNode && hoveredNode !== edge.to && hoveredNode !== edge.from ? 0.2 : 1}
            />
          ))}

          {modules.map(module => {
            const pos = positions[module.id];
            const color = categoryColors[module.category];
            const isSelected = selectedNode?.id === module.id;
            const isHovered = hoveredNode === module.id;
            const isConnected = hoveredNode && (
              module.dependencies.includes(hoveredNode) || 
              modules.find(m => m.id === hoveredNode)?.dependencies.includes(module.id)
            );
            const dimmed = hoveredNode && hoveredNode !== module.id && !isConnected;

            return (
              <g
                key={module.id}
                transform={`translate(${pos.x}, ${pos.y})`}
                onClick={() => setSelectedNode(selectedNode?.id === module.id ? null : module)}
                onMouseEnter={() => setHoveredNode(module.id)}
                onMouseLeave={() => setHoveredNode(null)}
                style={{ cursor: 'pointer' }}
                opacity={dimmed ? 0.25 : 1}
              >
                <rect
                  x="-52"
                  y="-18"
                  width="104"
                  height="36"
                  rx="6"
                  fill={color.bg}
                  stroke={isSelected ? '#fff' : isHovered ? '#e2e8f0' : color.border}
                  strokeWidth={isSelected ? 2.5 : isHovered ? 2 : 1.5}
                  filter={isSelected || isHovered ? 'url(#glow)' : ''}
                />
                
                <circle cx="42" cy="-10" r="9" fill="#1e293b" stroke={color.border} strokeWidth="1"/>
                <text x="42" y="-6" textAnchor="middle" fill="#e2e8f0" fontSize="8" fontWeight="bold">
                  {module.order}
                </text>

                <text x="0" y="4" textAnchor="middle" fill="white" fontSize="9" fontWeight="600">
                  {module.id.length > 12 ? module.id.substring(0, 11) + '..' : module.id}
                </text>
              </g>
            );
          })}
        </svg>
      </div>

      {selectedNode && (
        <div className="bg-slate-800 rounded-lg p-3 border border-slate-700 mb-3">
          <div className="flex items-start justify-between mb-3">
            <div>
              <div className="flex items-center gap-2 flex-wrap">
                <h2 className="text-base font-bold text-white">{selectedNode.id}</h2>
                <span className="px-2 py-0.5 rounded text-xs font-medium text-white" style={{ backgroundColor: categoryColors[selectedNode.category].bg }}>
                  {categoryColors[selectedNode.category].text}
                </span>
                <span className="px-2 py-0.5 rounded text-xs font-medium bg-slate-700 text-slate-300">
                  Order: {selectedNode.order}
                </span>
              </div>
              <p className="text-slate-400 text-xs mt-1">{selectedNode.description}</p>
            </div>
            <button onClick={() => setSelectedNode(null)} className="text-slate-400 hover:text-white text-lg ml-2">×</button>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-2">
            <div className="bg-slate-900 rounded p-2">
              <h3 className="text-green-400 font-semibold text-xs mb-1">→ Inputs</h3>
              <div className="space-y-0.5">
                {selectedNode.inputs.map((input, idx) => (
                  <div key={idx} className="text-slate-300 text-xs font-mono bg-slate-800 px-1.5 py-0.5 rounded">{input}</div>
                ))}
              </div>
            </div>

            <div className="bg-slate-900 rounded p-2">
              <h3 className="text-blue-400 font-semibold text-xs mb-1">← Outputs</h3>
              <div className="space-y-0.5">
                {selectedNode.outputs.map((output, idx) => (
                  <div key={idx} className="text-slate-300 text-xs font-mono bg-slate-800 px-1.5 py-0.5 rounded">{output}</div>
                ))}
              </div>
            </div>

            <div className="bg-slate-900 rounded p-2">
              <h3 className="text-orange-400 font-semibold text-xs mb-1">⬆ Dependencies</h3>
              {selectedNode.dependencies.length > 0 ? (
                <div className="flex flex-wrap gap-1">
                  {selectedNode.dependencies.map((dep, idx) => (
                    <span key={idx} className="text-xs px-1.5 py-0.5 rounded bg-orange-900/50 text-orange-300 cursor-pointer hover:bg-orange-800/50"
                      onClick={() => setSelectedNode(modules.find(m => m.id === dep))}>
                      {dep}
                    </span>
                  ))}
                </div>
              ) : (
                <span className="text-slate-500 text-xs">Root module</span>
              )}
            </div>

            <div className="bg-slate-900 rounded p-2">
              <h3 className="text-purple-400 font-semibold text-xs mb-1">★ Key Features</h3>
              {selectedNode.keyFeatures.map((f, idx) => (
                <div key={idx} className="text-slate-300 text-xs">• {f}</div>
              ))}
            </div>
          </div>

          <div className="mt-2 bg-slate-900 rounded p-2">
            <h3 className="text-red-400 font-semibold text-xs mb-1">⬇ Used By</h3>
            {(() => {
              const dependents = modules.filter(m => m.dependencies.includes(selectedNode.id));
              return dependents.length > 0 ? (
                <div className="flex flex-wrap gap-1">
                  {dependents.map((dep, idx) => (
                    <span key={idx} className="text-xs px-1.5 py-0.5 rounded bg-red-900/50 text-red-300 cursor-pointer hover:bg-red-800/50"
                      onClick={() => setSelectedNode(dep)}>
                      {dep.id}
                    </span>
                  ))}
                </div>
              ) : <span className="text-slate-500 text-xs">Terminal module</span>;
            })()}
          </div>
        </div>
      )}

      <div className="grid grid-cols-4 gap-2 mb-3">
        <div className="bg-slate-800 rounded p-2 text-center">
          <div className="text-xl font-bold text-white">{modules.length}</div>
          <div className="text-slate-400 text-xs">Modules</div>
        </div>
        <div className="bg-slate-800 rounded p-2 text-center">
          <div className="text-xl font-bold text-white">{edges.length}</div>
          <div className="text-slate-400 text-xs">Dependencies</div>
        </div>
        <div className="bg-slate-800 rounded p-2 text-center">
          <div className="text-xl font-bold text-white">5</div>
          <div className="text-slate-400 text-xs">Levels</div>
        </div>
        <div className="bg-slate-800 rounded p-2 text-center">
          <div className="text-xl font-bold text-white">40</div>
          <div className="text-slate-400 text-xs">Instructions</div>
        </div>
      </div>

      <div className="bg-slate-800 rounded-lg p-2">
        <h3 className="text-white font-semibold text-sm mb-2">Implementation Order</h3>
        <div className="flex flex-wrap gap-1 items-center">
          {modules.sort((a, b) => a.order - b.order).map((module, idx) => (
            <React.Fragment key={module.id}>
              <span 
                className="flex items-center gap-1 cursor-pointer hover:opacity-80 px-1.5 py-0.5 rounded"
                style={{ backgroundColor: categoryColors[module.category].bg + '33' }}
                onClick={() => setSelectedNode(module)}
              >
                <span className="w-5 h-5 rounded-full flex items-center justify-center text-xs font-bold text-white"
                  style={{ backgroundColor: categoryColors[module.category].bg }}>
                  {module.order}
                </span>
                <span className="text-slate-300 text-xs">{module.id}</span>
              </span>
              {idx < modules.length - 1 && <span className="text-slate-600 text-xs">→</span>}
            </React.Fragment>
          ))}
        </div>
      </div>
    </div>
  );
};

export default RISCV32IKnowledgeGraph;