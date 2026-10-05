# LangChain Agentic Workflow

A production-ready multi-step AI agent with tool calling, prompt orchestration, memory management, and evaluation harnesses.

## Features

- Multi-step agentic workflows with tool use
- Configurable prompt chains with LangChain
- Agent memory and context management
- Built-in tool registry with calculator, summarizer, web search
- Structured logging for agent decision tracing
- Modular tool registry — add custom tools easily

## Installation

pip install -r requirements.txt

## Quick Start

from agent import ResearchAgent

agent = ResearchAgent()
result = agent.run("What are the latest developments in LLM evaluation?")
print(result.final_answer)
print(result.steps_taken)

## Built-in Tools

- calculator — Perform mathematical calculations
- summarizer — Summarize long text passages
- web_search — Search for current information
- fact_checker — Validate factual claims

## Tech Stack

- Python, LangChain
- OpenAI API, Gemini API
- FastAPI, AWS, GCP, Azure