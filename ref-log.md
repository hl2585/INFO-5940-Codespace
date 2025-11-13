# Assignment 2 Reflection

## What I learned from implementing a multi-agent workflow

Building this assignment helped me see how multi-agent systems are really about separation of roles and clear contracts between agents. Designing distinct system prompts for the Planner and Reviewer forced me to think about what each agent is responsible for. The Planner focuses on generating a coherent, day-by-day itinerary from a vague user request, while the Reviewer acts like a critical editor that fact-checks, improves logistics, and communicates a final user-facing answer. I also learned how tool use fits into this workflow. The Planner works offline using only prior knowledge, while the Reviewer selectively calls `internet_search` to patch in reliable, up-to-date information.

## Challenges and how I addressed them

One challenge was figuring out how specific the system prompts needed to be to consistently produce structured outputs. My first drafts were too vague, and the agents sometimes skipped sections or mixed roles. I iterated on the wording to include explicit headings, bullet expectations, and examples of what to check (budget, logistics, opening hours). Another challenge was wiring environment variables correctly. I initially hit authentication errors when running the app; I resolved this by setting OPENAI_API_KEY.

## Creative ideas, variations, or design choices

For the Planner, I intentionally asked for assumptions to be made explicit in the “Trip Overview” so that the Reviewer and user can see what was inferred (season, airport, pace). For the Reviewer, I designed a “Delta List → Revised Itinerary → Notes for the Traveler” structure so the user can see both what changed and the improved final version, instead of just a corrected plan. I also liked the idea of requiring at least one `internet_search` call per review while still telling the agent to be selective and only call the tool when it meaningfully improves the plan.

## External tools and GenAI assistance

I used a ChatGPT assistant to help brainstorm and refine the Planner and Reviewer system prompts, to double-check that the `internet_search` tool was wired correctly, and to reason through authentication errors. I reviewed, edited, and tested all suggested prompts and code to ensure they matched the assignment requirements and my own understanding.

