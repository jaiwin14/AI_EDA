from typing import Dict, Any, Optional, List
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class SimplePipeline:
    """Simple orchestrator for running agents in sequence"""
    
    def __init__(self):
        self.agents = {}
        self.results = {}
        
    def add_agent(self, name: str, agent):
        """Add an agent to the pipeline"""
        self.agents[name] = agent
        
    def run(self, context, selected_agents: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Run the pipeline with selected agents
        
        Args:
            context: DatasetContext from ingestion
            selected_agents: List of agent names to run, or None for all
            
        Returns:
            Tuple: (results dictionary, updated context)
        """
        if selected_agents is None:
            selected_agents = list(self.agents.keys())
            
        results = {'pipeline_start': datetime.now()}
        
        for agent_name in selected_agents:
            if agent_name in self.agents:
                try:
                    logger.info(f"Running {agent_name}...")
                    start_time = datetime.now()
                    
                    # Run agent
                    agent = self.agents[agent_name]
                    if agent_name == 'ingest':
                        # Ingest agent updates context in place
                        result = agent.process(context.df, context.filename)
                        context = result  # Update context
                        results[agent_name] = {'status': 'success', 'context_updated': True}
                    else:
                        # Other agents return results
                        result = agent.process(context)
                        results[agent_name] = result
                        results[agent_name]['status'] = 'success'
                    
                    end_time = datetime.now()
                    results[agent_name]['duration'] = (end_time - start_time).total_seconds()
                    
                    logger.info(f"✅ {agent_name} completed in {results[agent_name]['duration']:.2f}s")
                    
                except Exception as e:
                    logger.error(f"❌ {agent_name} failed: {str(e)}")
                    results[agent_name] = {
                        'status': 'error',
                        'error': str(e),
                        'duration': 0
                    }
                    
        results['pipeline_end'] = datetime.now()
        results['total_duration'] = (results['pipeline_end'] - results['pipeline_start']).total_seconds()
        
        return results, context

# Global pipeline instance
pipeline = SimplePipeline()
