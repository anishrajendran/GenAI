# filename: analyze_gpt_domains.py

from collections import defaultdict
from typing import Dict
import matplotlib.pyplot as plt

# Sample summaries of papers related to GPT models
summaries = [
    "As a pivotal extension of the renowned ChatGPT, the GPT Store serves as a dynamic marketplace for various Generative Pre-trained Transformer (GPT) models, shaping the frontier of conversational AI. This paper presents an in-depth measurement study of the GPT Store, with a focus on the categorization of GPTs by topic, factors influencing GPT popularity, and the potential security risks.",
    "This paper explores the use of Generative Pre-trained Transformers (GPT) in strategic game experiments, specifically the ultimatum game and the prisoner's dilemma.",
    "In this paper, we evaluate different abilities of GPT-4V including visual understanding, language understanding, visual puzzle solving, and understanding of other modalities such as depth, thermal, video, and audio.",
    "The mainstream BERT/GPT model contains only 10 to 20 layers, and there is little literature to discuss the training of deep BERT/GPT.",
    "In November 2023, OpenAI introduced a new service allowing users to create custom versions of ChatGPT (GPTs) by using specific instructions and knowledge to guide the model's behavior.",
    "LLMs have long demonstrated remarkable effectiveness in automatic program repair (APR), with OpenAI's ChatGPT being one of the most widely used models in this domain.",
    "This paper extends the concept of generalized polarization tensors (GPTs), which was previously defined for inclusions with homogeneous conductivities, to inhomogeneous conductivity inclusions.",
    "Auto-GPT is an autonomous agent that leverages recent advancements in adapting Large Language Models (LLMs) for decision-making tasks.",
    "Addressing the challenge of generating personalized feedback for programming assignments is demanding due to several factors, like the complexity of code syntax or different ways to correctly solve a task.",
    "This study presents a thorough examination of various Generative Pretrained Transformer (GPT) methodologies in sentiment analysis, specifically in the context of Task 4 on the SemEval 2017 dataset.",
    "The widespread adoption of the large language model (LLM), e.g. Generative Pre-trained Transformer (GPT), deployed on cloud computing environment (e.g. Azure) has led to a huge increased demand for resources.",
    "This work investigates two strategies for zero-shot non-intrusive speech assessment leveraging large language models.",
    "The recent release of GPT-4o has garnered widespread attention due to its powerful general capabilities.",
    "Radiotherapy treatment planning is a time-consuming and potentially subjective process that requires the iterative adjustment of model parameters to balance multiple conflicting objectives.",
    "The emergence of large language models (LLMs) has significantly accelerated the development of a wide range of applications across various fields.",
    "This paper explores the use of Large Language Models (LLMs) as decision aids, with a focus on their ability to learn preferences and provide personalized recommendations.",
    "The rapid advancements in Large Language Models (LLMs) have revolutionized natural language processing, with GPTs, customized versions of ChatGPT available on the GPT Store, emerging as a prominent technology for specific domains and tasks.",
    "The formalism of generalized probabilistic theories (GPTs) was originally developed as a way to characterize the landscape of conceivable physical theories.",
    "Generative Pre-trained Transformer (GPT) models have shown remarkable capabilities for natural language generation, but their performance for machine translation has not been thoroughly investigated.",
    "Background: Depression is a common mental disorder with societal and economic burden.",
    "Generative pre-trained transformer (GPT) models have revolutionized the field of natural language processing (NLP) with remarkable performance in various tasks and also extend their power to multimodal domains.",
    "Decoder-only Transformer models such as GPT have demonstrated exceptional performance in text generation, by autoregressively predicting the next token.",
    "Significant advancements have recently been made in large language models represented by GPT models.",
    "Following OpenAI's introduction of GPTs, a surge in GPT apps has led to the launch of dedicated LLM app stores.",
    "The United States has experienced a significant increase in violent extremism, prompting the need for automated tools to detect and limit the spread of extremist ideology online.",
    "This study aims to investigate whether GPT-4 can effectively grade assignments for design university students and provide useful feedback.",
    "Creating specialized large language models requires vast amounts of clean, special purpose data for training and fine-tuning.",
    "Scoring student-drawn models is time-consuming.",
    "The study evaluates and compares GPT-4 and GPT-4Vision for radiological tasks, suggesting GPT-4Vision may recognize radiological features from images, thereby enhancing its diagnostic potential over text-based descriptions.",
    "We introduce the framework of general probabilistic theories (GPTs for short).",
    "As digital healthcare evolves, the security of electronic health records (EHR) becomes increasingly crucial.",
    "Large language models (LLMs) have achieved impressive results across various tasks.",
    "Generative Pre-Trained Transformers (GPTs) are hyped to revolutionize robotics.",
    "Large multimodal models (LMMs) extend large language models (LLMs) with multi-sensory skills, such as visual understanding, to achieve stronger generic intelligence.",
    "The debate around the use of GPT 3.5 has been a popular topic among academics since the release of ChatGPT.",
    "This paper presents an empirical evaluation of the performance of the Generative Pre-trained Transformer (GPT) model in Harvard's CS171 data visualization course.",
    "Anomaly detection is a crucial task across different domains and data types.",
    "Due to the lack of a large collection of high-quality labeled sentence pairs with textual similarity scores, existing approaches for Semantic Textual Similarity (STS) mostly rely on unsupervised techniques or training signals that are only partially correlated with textual similarity.",
    "In this study, we introduce T2M-HiFiGPT, a novel conditional generative framework for synthesizing human motion from textual descriptions.",
    "Large language models (LLMs) have demonstrated a powerful ability to answer various queries as a general-purpose assistant.",
    "OpenAI's ChatGPT initiated a wave of technical iterations in the space of Large Language Models (LLMs) by demonstrating the capability and disruptive power of LLMs.",
    "Recently, the concept of artificial assistants has evolved from science fiction into real-world applications.",
    "A conductivity inclusion, inserted in a homogeneous background, induces a perturbation in the background potential.",
    "Language models (LMs) pre-trained on massive amounts of text, in particular bidirectional encoder representations from Transformers (BERT), generative pre-training (GPT), and GPT-2, have become a key technology for many natural language processing tasks.",
    "As large language models (LLMs) like GPT become increasingly prevalent, it is essential that we assess their capabilities beyond language processing.",
    "Generative, pre-trained transformers (GPTs, a.k.a. 'Foundation Models') have reshaped natural language processing (NLP) through their versatility in diverse downstream tasks.",
    "Autonomous driving technology is poised to transform transportation systems.",
    "This paper introduces the 'GPT-in-the-loop' approach, a novel method combining the advanced reasoning capabilities of Large Language Models (LLMs) like Generative Pre-trained Transformers (GPT) with multiagent (MAS) systems.",
    "The integration of robots in chemical experiments has enhanced experimental efficiency, but lacking the human intelligence to comprehend literature, they seldom provide assistance in experimental design.",
    "This study explores the capabilities of prompt-driven Large Language Models (LLMs) like ChatGPT and GPT-4 in adhering to human guidelines for dialogue summarization."
]

# Define keywords for each domain
domain_keywords = {
    "Security and Privacy": ["security", "privacy", "safety", "risk"],
    "Social Science and Psychology": ["social", "psychology", "sentiment", "mental health", "depression"],
    "Natural Language Processing (NLP)": ["language", "translation", "NLP", "dialogue", "text"],
    "Healthcare and Medicine": ["healthcare", "medicine", "medical", "radiotherapy", "depression"],
    "Education and Learning": ["education", "learning", "grading", "feedback"],
    "Robotics and Automation": ["robotics", "autonomous", "automation"],
    "Data Science and Machine Learning": ["data", "machine learning", "model", "evaluation"],
    "Other Domains": []
}

# Categorize papers into domains
domain_counts = defaultdict(int)

for summary in summaries:
    categorized = False
    for domain, keywords in domain_keywords.items():
        if any(keyword in summary.lower() for keyword in keywords):
            domain_counts[domain] += 1
            categorized = True
            break
    if not categorized:
        domain_counts["Other Domains"] += 1

# Generate bar chart
def generate_bar_chart(domains: Dict[str, int], output_file: str) -> None:
    fig, ax = plt.subplots()
    ax.bar(domains.keys(), domains.values())
    plt.xticks(rotation=45, ha="right")
    plt.xlabel("Application Domains")
    plt.ylabel("Number of Papers")
    plt.title("Number of Papers per Application Domain")

    plt.tight_layout()
    plt.savefig(output_file)
    plt.show()

# Generate and save the bar chart
generate_bar_chart(domain_counts, "gpt_application_domains.png")