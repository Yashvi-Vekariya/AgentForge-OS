import logging
import asyncio
from typing import Dict, Any, List, Optional
from concurrent.futures import ThreadPoolExecutor
from crewai import Agent, Task, Crew, Process
from app.agents.agent_factory import AgentFactory
from app.llm_manager import LLMManager
from app.memory.memory_manager import MemoryManager

logger = logging.getLogger(__name__)

class AdvancedCrewOrchestrator:
    """Advanced CrewAI orchestrator with parallel execution and error handling"""
    
    def __init__(self, 
                 llm_manager: LLMManager,
                 memory_manager: MemoryManager,
                 agent_factory: AgentFactory):
        self.llm_manager = llm_manager
        self.memory_manager = memory_manager
        self.agent_factory = agent_factory
        self.executor = ThreadPoolExecutor(max_workers=10)
        self.crew = None
        
    def create_sequential_workflow(self, 
                                 workflow_config: Dict[str, Any]) -> Crew:
        """Create sequential workflow with multiple agents"""
        agents = []
        tasks = []
        
        for agent_config in workflow_config.get('agents', []):
            agent_type = agent_config['type']
            agent_name = agent_config.get('name', agent_type)
            
            # Create CrewAI agent
            crew_agent = Agent(
                role=agent_config.get('role', ''),
                goal=agent_config.get('goal', ''),
                backstory=agent_config.get('backstory', ''),
                llm=self.llm_manager.get_crewai_llm(),
                allow_delegation=agent_config.get('allow_delegation', False),
                verbose=True
            )
            agents.append(crew_agent)
            
            # Create task for this agent
            task = Task(
                description=agent_config.get('task', ''),
                agent=crew_agent,
                expected_output=agent_config.get('expected_output', '')
            )
            tasks.append(task)
        
        # Create crew with sequential process
        self.crew = Crew(
            agents=agents,
            tasks=tasks,
            process=Process.sequential,
            verbose=True,
            memory=True
        )
        
        return self.crew
    
    async def execute_workflow_async(self, 
                                   workflow_config: Dict[str, Any],
                                   inputs: Dict[str, Any] = None) -> Dict[str, Any]:
        """Execute workflow asynchronously"""
        try:
            loop = asyncio.get_event_loop()
            
            # Run crew execution in thread pool
            crew = self.create_sequential_workflow(workflow_config)
            result = await loop.run_in_executor(
                self.executor, 
                crew.kickoff,
                inputs
            )
            
            return {
                "workflow_result": result,
                "status": "completed",
                "workflow_type": "sequential"
            }
            
        except Exception as e:
            logger.error(f"Error in async workflow execution: {str(e)}")
            return {
                "workflow_result": None,
                "status": "failed",
                "error": str(e)
            }
    
    def execute_parallel_tasks(self, 
                             tasks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Execute multiple agent tasks in parallel"""
        try:
            futures = []
            task_results = []
            
            for task_config in tasks:
                agent_type = task_config['agent_type']
                task_description = task_config['task']
                context = task_config.get('context', {})
                
                # Create agent and submit task
                agent = self.agent_factory.create_agent(agent_type)
                future = self.executor.submit(agent.act, task_description, context)
                futures.append((agent_type, task_description, future))
            
            # Collect results
            for agent_type, task_description, future in futures:
                try:
                    result = future.result(timeout=300)  # 5-minute timeout
                    task_results.append({
                        "agent_type": agent_type,
                        "task": task_description,
                        "result": result,
                        "status": "completed"
                    })
                except Exception as e:
                    task_results.append({
                        "agent_type": agent_type,
                        "task": task_description,
                        "result": None,
                        "status": "failed",
                        "error": str(e)
                    })
            
            return {
                "parallel_results": task_results,
                "successful_tasks": len([r for r in task_results if r['status'] == 'completed']),
                "failed_tasks": len([r for r in task_results if r['status'] == 'failed']),
                "status": "completed"
            }
            
        except Exception as e:
            logger.error(f"Error in parallel execution: {str(e)}")
            return {
                "parallel_results": [],
                "successful_tasks": 0,
                "failed_tasks": 0,
                "status": "failed",
                "error": str(e)
            }
    
    def create_conditional_workflow(self, 
                                  workflow_config: Dict[str, Any]) -> Dict[str, Any]:
        """Create workflow with conditional execution paths"""
        try:
            main_agents = []
            conditional_branches = []
            
            # Create main sequence agents
            for agent_config in workflow_config.get('main_sequence', []):
                agent = self.agent_factory.create_agent(agent_config['type'])
                main_agents.append({
                    "agent": agent,
                    "config": agent_config
                })
            
            # Create conditional branches
            for branch_config in workflow_config.get('conditional_branches', []):
                condition = branch_config['condition']
                branch_agents = []
                
                for agent_config in branch_config['agents']:
                    agent = self.agent_factory.create_agent(agent_config['type'])
                    branch_agents.append({
                        "agent": agent,
                        "config": agent_config
                    })
                
                conditional_branches.append({
                    "condition": condition,
                    "agents": branch_agents
                })
            
            return {
                "main_agents": main_agents,
                "conditional_branches": conditional_branches,
                "workflow_type": "conditional"
            }
            
        except Exception as e:
            logger.error(f"Error creating conditional workflow: {str(e)}")
            return {
                "error": str(e),
                "status": "failed"
            }
    
    def execute_with_retry(self, 
                         agent_type: str, 
                         task: str, 
                         context: Dict[str, Any] = None,
                         max_retries: int = 3) -> Dict[str, Any]:
        """Execute agent task with retry logic"""
        last_error = None
        
        for attempt in range(max_retries):
            try:
                agent = self.agent_factory.create_agent(agent_type)
                result = agent.act(task, context)
                
                if result.get('status') == 'completed':
                    return {
                        **result,
                        "attempts": attempt + 1,
                        "retry_used": attempt > 0
                    }
                
            except Exception as e:
                last_error = e
                logger.warning(f"Attempt {attempt + 1} failed for {agent_type}: {str(e)}")
                
                # Wait before retry (exponential backoff)
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt
                    import time
                    time.sleep(wait_time)
        
        return {
            "agent": agent_type,
            "task": task,
            "response": f"Failed after {max_retries} attempts: {str(last_error)}",
            "status": "failed",
            "error": str(last_error),
            "attempts": max_retries
        }