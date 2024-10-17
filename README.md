# Awesome-LLM-Jailbreak

## Awesome LLM Jailbreak

Welcome to the **Awesome LLM Jailbreak** repository! This project curates a list of high-quality resources related to LLM Jailbreak, including research papers, tools, libraries, and more.

## Daily Updates

### 2024-10-17

### 1. [Negative-Prompt-driven Alignment for Generative Language Model](https://arxiv.org/pdf/2410.12194)

**Summary**: The paper introduces NEAT, a method that uses negative prompts to align generative language models with human values by penalizing undesirable outputs. By incorporating both positive and negative examples into the training process, NEAT improves the model's ability to avoid harmful or biased responses, enhancing overall alignment with human preferences.

### 2. [Iter-AHMCL: Alleviate Hallucination for Large Language Model via Iterative Model-level Contrastive Learning](https://arxiv.org/pdf/2410.12130)

**Summary**: The paper introduces Iterative Model-level Contrastive Learning (Iter-AHMCL) to reduce hallucinations in Large Language Models (LLMs) by modifying representation layers using contrastive models trained on data with and without hallucinations. This approach achieves an average improvement of 10.1 points on the TruthfulQA benchmark across four LLMs, demonstrating its effectiveness in reducing hallucination while preserving model capabilities.

### 3. [Exploring Large Language Models for Hate Speech Detection in Rioplatense Spanish](https://arxiv.org/pdf/2410.12174)

**Summary**: The paper investigates the performance of LLMs like ChatGPT 3.5, Mixtral, and Aya in detecting hate speech in Rioplatense Spanish, comparing them to a fine-tuned BERT classifier. While LLMs exhibit lower precision and may struggle with certain slang or slurs, they demonstrate sensitivity to nuanced hate speech, particularly homophobic and transphobic content. The study emphasizes the importance of tailored corpora for effective hate speech detection and makes its resources available for further research.

### 4. [Semantics-Adaptive Activation Intervention for LLMs via Dynamic Steering Vectors](https://arxiv.org/pdf/2410.12299)

**Summary**: The paper introduces Semantics-Adaptive Dynamic Intervention (SADI), a novel method for modifying the behavior of LLMs by dynamically adjusting activation vectors based on input semantics. SADI outperforms existing methods by identifying and scaling critical model elements during inference, enhancing task performance without additional training, and demonstrating versatility across different LLM architectures and tasks.

### 5. [Open Domain Question Answering with Conflicting Contexts](https://arxiv.org/pdf/2410.12311)

**Summary**: The paper introduces the Question Answering with Conflicting Contexts (QACC) dataset, highlighting that up to 25% of open domain questions yield conflicting information when retrieved from sources like Google Search. The study evaluates three Large Language Models (LLMs) and shows their limitations in handling such conflicts. By fine-tuning LLMs to explain their answers, the authors suggest a potential improvement in their ability to reason through conflicting contexts.

### 6. [Neuron-based Personality Trait Induction in Large Language Models](https://arxiv.org/pdf/2410.12327)

**Summary**: The paper introduces a neuron-based approach for inducing personality traits in LLMs, focusing on the Big Five personality traits. It contributes by creating PersonalityBench, a dataset for evaluating LLM personality traits, and proposes methods to identify and manipulate personality-related neurons within LLMs. This approach allows for fine-grained control over LLM traits without retraining, achieving performance comparable to fine-tuned models.

### 7. [A linguistic analysis of undesirable outcomes in the era of generative AI](https://arxiv.org/pdf/2410.12341)

**Summary**: The paper examines the linguistic degradation in generative AI models, particularly LLama2, showing that repeated fine-tuning on self-generated content leads to reduced lexical richness and distorted linguistic patterns. This "model collapse" not only diminishes content diversity but also introduces inaccuracies and biases, emphasizing the need for careful input curation to mitigate these issues.

### 8. [ProSA: Assessing and Understanding the Prompt Sensitivity of LLMs](https://arxiv.org/pdf/2410.12405)

**Summary**: The paper introduces ProSA, a framework for assessing and understanding the prompt sensitivity of LLMs. It introduces a new metric, PromptSensiScore, and uses decoding confidence to analyze how prompt variations affect model performance and subjective evaluations. The study finds that prompt sensitivity varies across datasets and models, with larger models showing greater robustness, and that few-shot examples can mitigate sensitivity issues, especially in complex tasks.

### 9. [Conformity in Large Language Models](https://arxiv.org/pdf/2410.12428)

**Summary**: The paper investigates the conformity bias in LLMs, finding that all tested models exhibit varying levels of conformity to majority opinions, especially when uncertain about their own predictions. The study also identifies factors influencing conformity, such as training paradigms and input characteristics, and proposes interventions like Devil's Advocate and Question Distillation to reduce this bias in LLMs.

### 10. [Retrieval-Reasoning Large Language Model-based Synthetic Clinical Trial Generation](https://arxiv.org/pdf/2410.12476)

**Summary**: The paper introduces a Retrieval-Reasoning framework using LLMs to generate synthetic clinical trials, addressing challenges like data scarcity and ethical concerns. The synthetic trials, with binary success/failure labels, effectively augment real datasets, improving model training for trial outcome prediction. This approach shows potential for accelerating clinical research while maintaining patient privacy.

### 11. [With a Grain of SALT: Are LLMs Fair Across Social Dimensions?](https://arxiv.org/pdf/2410.12499)

**Summary**: The paper investigates biases in open-source Large Language Models (LLMs) across gender, religion, and race by introducing a bias detection dataset generated using seven bias triggers. The study evaluates Llama and Gemma models, finding consistent polarization toward certain groups, with language variations revealing cultural and contextual influences on bias manifestation.

### 12. [Advancing Fairness in Natural Language Processing: From Traditional Methods to Explainability](https://arxiv.org/pdf/2410.12511)

**Summary**: This PhD thesis explores the integration of fairness and explainability in Natural Language Processing (NLP), introducing innovative algorithms to mitigate biases in multi-class classifiers and Transformer models. It also critiques standard fairness metrics and proposes new methods like COCKATIEL and TaCo to enhance transparency and equity in NLP systems, contributing to the broader discourse on responsible AI development.

### 13. [Insights from the Inverse: Reconstructing LLM Training Goals Through Inverse RL](https://arxiv.org/pdf/2410.12491)

**Summary**: The paper presents a novel approach to interpreting LLMs trained with Reinforcement Learning from Human Feedback (RLHF) by applying inverse reinforcement learning (IRL) to recover their implicit reward functions. The study finds that IRL-derived reward models can predict human preferences with high accuracy and be used to fine-tune new LLMs, offering insights into the non-identifiability of reward functions and the relationship between model size and interpretability. This work enhances understanding of LLM alignment and has implications for the responsible development of these models.

### 14. [On the Risk of Evidence Pollution for Malicious Social Text Detection in the Era of LLMs](https://arxiv.org/pdf/2410.12600)

**Summary**: The paper investigates the risk of evidence pollution in malicious social text detection due to the rise of LLMs, which can manipulate evidence to confuse detectors. It proposes three defense strategies to mitigate this risk but highlights practical limitations, such as the need for annotated data and high inference costs. The study concludes that polluted evidence, especially when generated by LLMs, significantly compromises detection models and can amplify negative impacts.

### 15. [Can We Reverse In-Context Knowledge Edits?](https://arxiv.org/pdf/2410.12586)

**Summary**: The paper investigates the detection and reversal of in-context knowledge edits (IKE) in LLMs, which can be misused to insert harmful content. The authors demonstrate high accuracy in detecting IKE-edits using top-10 token probabilities and introduce a method to reverse these edits using specially tuned reversal tokens, enhancing LLM resilience and transparency.

### 16. [Weak-to-Strong Generalization beyond Accuracy: a Pilot Study in Safety, Toxicity, and Legal Reasoning](https://arxiv.org/pdf/2410.12621)

**Summary**: The paper explores the application of weak-to-strong generalization in practical alignment tasks such as safety, toxicity, and legal reasoning, addressing the limitations of traditional human feedback-based methods for superhuman language models. It demonstrates the effectiveness of this approach in complex alignment scenarios and proposes strategies to improve alignment performance, aiming to advance research in this area.

### 17. [Building Better: Avoiding Pitfalls in Developing Language Resources when Data is Scarce](https://arxiv.org/pdf/2410.12691)

**Summary**: The paper examines the challenges of developing language resources for mid- to low-resource languages, focusing on issues related to data quality and ethical annotation practices. Through feedback from stakeholders, it identifies concerns such as linguistic and cultural appropriateness of data and the misuse of online communities for annotation. The study concludes with recommendations for creating high-quality, culturally sensitive language resources that respect the dignity of data workers.

### 18. [Unitary Multi-Margin BERT for Robust Natural Language Processing](https://arxiv.org/pdf/2410.12759)

**Summary**: The paper introduces UniBERT, a novel approach to enhance the robustness of BERT models against adversarial attacks by combining unitary weights with multi-margin loss. This method significantly improves post-attack classification accuracy by up to 73.8% while maintaining competitive pre-attack performance, with the tradeoff between pre- and post-attack accuracy adjustable via a single parameter.

### 19. [WorldMedQA-V: a multilingual, multimodal medical examination dataset for multimodal language models evaluation](https://arxiv.org/pdf/2410.12722)

**Summary**: The paper introduces WorldMedQA-V, a multilingual, multimodal medical examination dataset designed to evaluate vision language models (VLMs) in healthcare. The dataset includes 568 multiple-choice questions paired with medical images from four countries, with questions available in both original languages and validated English translations. The authors provide baseline performance metrics for various models, highlighting the importance of this benchmark for ensuring the safety and efficacy of VLMs in diverse healthcare settings.

### 20. [Boosting Logical Fallacy Reasoning in LLMs via Logical Structure Tree](https://arxiv.org/pdf/2410.12048)

**Summary**: The paper introduces a novel approach to enhance logical fallacy reasoning in LLMs by constructing a logical structure tree that explicitly represents the hierarchical logic flow within statements. This tree, built using unsupervised methods and guided by constituency trees and a taxonomy of connectives, is integrated into LLMs through both textual and embedding-based prompts. The approach significantly improves the precision and recall of fallacy detection and classification in benchmark datasets.

### 21. [Concept-Reversed Winograd Schema Challenge: Evaluating and Improving Robust Reasoning in Large Language Models via Abstraction](https://arxiv.org/pdf/2410.12040)

**Summary**: The paper introduces the Concept-Reversed Winograd Schema Challenge (CR-WSC) to test the robustness of Large Language Models (LLMs) in reasoning, finding that reversing concepts associated with wrong answers significantly reduces their performance. Additionally, the authors propose Abstraction-of-Thought (AoT), a method that uses conceptual abstraction to enhance LLMs' consistency and robustness in reasoning, as evidenced by experiments on the CR-WSC dataset.

### 22. [Bias Similarity Across Large Language Models](https://arxiv.org/pdf/2410.12010)

**Summary**: The paper investigates bias similarity across ten Large Language Models (LLMs) from four model families, finding that fine-tuning does not effectively mitigate bias, models within the same family exhibit dissimilar output distributions, and there is a risk of training data information leakage. The study highlights the challenges in addressing bias in LLMs and raises concerns about their real-world deployment.

### 23. [Preference Optimization with Multi-Sample Comparisons](https://arxiv.org/pdf/2410.12138)

**Summary**: The paper introduces a new approach to post-training optimization for generative models, specifically addressing the limitations of single-sample comparisons in methods like RLHF and DAP. By proposing Multi-sample Direct Preference Optimization (mDPO) and Multi-sample Identity Preference Optimization (mIPO), the authors demonstrate that multi-sample comparisons are more effective in capturing generative diversity and bias, leading to a more robust optimization framework, especially in the presence of label noise.

### 24. [Controlled Automatic Task-Specific Synthetic Data Generation for Hallucination Detection](https://arxiv.org/pdf/2410.12278)

**Summary**: The paper introduces a novel method for generating task-specific synthetic datasets for hallucination detection, featuring a two-step pipeline with hallucination pattern guidance and language style alignment. The approach improves the generalization and robustness of hallucination detectors, outperforming in-context-learning-based detectors by 32%. The data mixture strategy further enhances the performance across different tasks and generators.

### 25. [OmnixR: Evaluating Omni-modality Language Models on Reasoning across Modalities](https://arxiv.org/pdf/2410.12219)

**Summary**: The paper introduces OmnixR, an evaluation suite for benchmarking Omni-modality Language Models (OLMs) like GPT-4o and Gemini, which are designed to handle multiple modalities (text, vision, audio). OmnixR addresses the lack of comprehensive multi-modal assessments by offering synthetic and realistic subsets, challenging models to reason across diverse modalities. The study reveals that current state-of-the-art OLMs struggle with tasks requiring integrated multi-modal reasoning, highlighting the need for improved alignment in omni-modal AI systems.

### 26. [Cross-Modal Safety Mechanism Transfer in Large Vision-Language Models](https://arxiv.org/pdf/2410.12662)

**Summary**: The paper identifies a vulnerability in Large Vision-Language Models (LVLMs) where the safety mechanisms designed for text are not effectively transferred to visual inputs. The authors propose a Text-Guided vision-language Alignment method (TGA) that uses related text to guide the projection of visual data into the hidden states space, ensuring the safety mechanism is correctly applied to images. This approach maintains both safety and performance across various vision tasks.

### 27. [CREAM: Consistency Regularized Self-Rewarding Language Models](https://arxiv.org/pdf/2410.12735)

**Summary**: The paper introduces CREAM, a Consistency Regularized Self-Rewarding Language Model, to address the issue of accumulated bias in self-rewarding language models, which can lead to unreliable preference data. By leveraging rewarding consistency across iterations, CREAM helps the model learn from more reliable data, resulting in improved reward consistency and alignment performance.

### 28. [TaCo: Targeted Concept Erasure Prevents Non-Linear Classifiers From Detecting Protected Attributes](https://arxiv.org/pdf/2312.06499)

**Summary**: The paper introduces Targeted Concept Erasure (TaCo), a method designed to remove sensitive information from latent representations in NLP models, ensuring fairness against non-linear classifiers. TaCo outperforms existing methods by significantly reducing the prediction accuracy of sensitive attributes while maintaining overall task performance.

### 29. [UNDIAL: Self-Distillation with Adjusted Logits for Robust Unlearning in Large Language Models](https://arxiv.org/pdf/2402.10052)

**Summary**: The paper introduces UnDIAL, a robust unlearning method for large language models that uses self-distillation with adjusted logits to selectively reduce the influence of targeted tokens, thereby enhancing privacy and safety. Unlike existing methods, UnDIAL ensures stable convergence and avoids over-unlearning, demonstrating robustness and scalability across various unlearning tasks.

### 30. [ToBlend: Token-Level Blending With an Ensemble of LLMs to Attack AI-Generated Text Detection](https://arxiv.org/pdf/2402.11167)

**Summary**: The paper introduces ToBlend, a token-level ensemble method that combines outputs from multiple LLMs to generate text that can evade AI-content detection systems. By randomly selecting tokens from different LLMs, ToBlend significantly reduces the accuracy of current detection methods. The study also fine-tunes a Llama3.1 model to better identify ToBlend-generated text, highlighting the potential of such adversarial techniques to improve detection robustness.

### 31. [Meta-Unlearning on Diffusion Models: Preventing Relearning Unlearned Concepts](https://arxiv.org/pdf/2410.12777)

**Summary**: The paper introduces meta-unlearning for diffusion models (DMs) to prevent the relearning of unlearned harmful or copyrighted concepts through malicious finetuning. The proposed method ensures that benign concepts related to unlearned ones self-destruct when exposed to such finetuning, thereby maintaining the integrity of the unlearning process. The approach is validated on Stable Diffusion models and is shown to be effective and compatible with existing unlearning techniques.

### 32. [A Watermark for Low-entropy and Unbiased Generation in Large Language Models](https://arxiv.org/pdf/2405.14604)

**Summary**: The paper introduces the Sampling One Then Accepting (STA-1) method, a novel watermarking technique for LLMs that addresses several limitations of previous unbiased watermarking methods, including the need for white-box access, long detection times, and lack of robustness against attacks. The study highlights the tradeoff between watermark strength and text quality, particularly in low-entropy scenarios, and demonstrates that STA-1 achieves comparable text quality and watermark strength with a reduced risk of unsatisfactory outputs.

### 33. [The Comparative Trap: Pairwise Comparisons Amplifies Biased Preferences of LLM Evaluators](https://arxiv.org/pdf/2406.12319)

**Summary**: The paper highlights that LLM evaluators exhibit biased preferences, particularly when using pairwise comparisons, which amplify superficial attributes like verbosity. To mitigate this, the authors propose PRePair, a method that combines pointwise reasoning within a pairwise framework, enhancing unbiased evaluation and improving performance on both adversarial and standard benchmarks.

### 34. [MFC-Bench: Benchmarking Multimodal Fact-Checking with Large Vision-Language Models](https://arxiv.org/pdf/2406.11288)

**Summary**: The paper introduces MFC-Bench, a benchmark for evaluating the factual accuracy of large vision-language models (LVLMs) in multimodal fact-checking. It assesses models across three stages of verdict prediction and reveals that current LVLMs struggle with detecting manipulated content, highlighting the need for more reliable AI systems. The benchmark and resources are made publicly available to support further research in this area.

### 35. [CELL your Model: Contrastive Explanations for Large Language Models](https://arxiv.org/pdf/2406.11785)

**Summary**: The paper introduces contrastive explanation methods for LLMs by identifying why an LLM produces a specific output in response to a prompt. The authors propose two algorithms: a myopic algorithm that requires many model calls and a budgeted algorithm that efficiently adheres to a query budget, both of which are shown to be effective across various natural language tasks.

### 36. [Exploring Changes in Nation Perception with Nationality-Assigned Personas in LLMs](https://arxiv.org/pdf/2406.13993)

**Summary**: The study investigates how Large Language Models (LLMs) alter their perceptions of nations when assigned specific nationality personas, finding a bias favoring Western European nations. The research highlights discrepancies between LLM evaluations and human survey responses, emphasizing the need for mechanisms to ensure fairness and prevent over-generalization in LLM outputs.

### 37. [Enhancing Data Privacy in Large Language Models through Private Association Editing](https://arxiv.org/pdf/2406.18221)

**Summary**: The paper introduces Private Association Editing (PAE), a novel method to enhance data privacy in LLMs by removing Personally Identifiable Information (PII) without retraining the model. Experimental results show that PAE outperforms alternative methods, suggesting it as a crucial tool for protecting data privacy in LLMs, thereby fostering safer model development for practical applications.

### 38. [Core: Robust Factual Precision with Informative Sub-Claim Identification](https://arxiv.org/pdf/2407.03572)

**Summary**: The paper introduces Core, a customizable subclaim selection component designed to enhance the robustness of factual precision metrics in large language models by filtering out repetitive and non-informative subclaims. The authors demonstrate that integrating Core with existing metrics significantly improves their performance across various knowledge domains and advocate for its adoption in the community, releasing an expanded dataset and an evaluation framework to support further research.

### 39. [Measuring and Benchmarking Large Language Models' Capabilities to Generate Persuasive Language](https://arxiv.org/pdf/2406.17753)

**Summary**: The paper investigates the capability of Large Language Models (LLMs) to generate persuasive language across various domains, comparing their performance when explicitly instructed to enhance or reduce persuasion and when merely paraphrasing. The study introduces a new dataset, Persuasive-Pairs, which pairs original texts with LLM-generated rewrites, annotated on a relative scale for persuasion. The analysis reveals that different personas in LLaMA3's system prompt significantly influence the level of persuasive language, even in paraphrasing tasks.

### 40. [Beyond Instruction Following: Evaluating Inferential Rule Following of Large Language Models](https://arxiv.org/pdf/2407.08440)

**Summary**: The paper introduces RuleBench, a benchmark to evaluate the inferential rule-following capabilities of Large Language Models (LLMs), distinguishing it from mere instruction-following. The study finds that LLMs struggle with rule-following and proposes Inferential Rule-Following Tuning (IRFT) to improve this ability, showing that LLMs can learn abstract rule-following from synthetic data and apply it to RuleBench scenarios.

### 41. [LoraMap: Harnessing the Power of LoRA Connections](https://arxiv.org/pdf/2408.16264)

**Summary**: The paper introduces LoraMap, a novel approach for connecting multiple Low-Rank Adaptation (LoRA) modules in Large Language Models (LLMs) to enhance fact-checking performance. By creating specialized reasoning datasets and fine-tuning individual LoRAs, LoraMap establishes connections between these modules, outperforming existing methods like LoraHub and LoraConcat in fact-checking tasks while using fewer trainable parameters.

### 42. [MMLU-Pro+: Evaluating Higher-Order Reasoning and Shortcut Learning in LLMs](https://arxiv.org/pdf/2409.02257)

**Summary**: The paper introduces MMLU-Pro+, an advanced benchmark designed to evaluate higher-order reasoning and shortcut learning in LLMs. By including questions with multiple correct answers across various domains, MMLU-Pro+ challenges LLMs to engage in complex reasoning and resist simplistic problem-solving strategies. The study reveals significant performance variations among state-of-the-art LLMs, emphasizing the need for more rigorous evaluation frameworks.

### 43. [I Want to Break Free! Persuasion and Anti-Social Behavior of LLMs in Multi-Agent Settings with Social Hierarchy](https://arxiv.org/pdf/2410.07109)

**Summary**: The paper investigates the interactions of Large Language Model (LLM) agents in a simulated social hierarchy, inspired by the Stanford Prison Experiment, focusing on persuasion and anti-social behavior. It finds that the goal of an agent significantly affects its persuasiveness but not its anti-social behavior, and that agents' personas, particularly the guard's personality, influence both persuasion success and anti-social behavior. The study also reveals that assigning roles alone can lead to anti-social behavior without explicit personality prompts, highlighting implications for LLM development and societal impact.

### 44. [MarkLLM: An Open-Source Toolkit for LLM Watermarking](https://arxiv.org/pdf/2405.10051)

**Summary**: The paper introduces MarkLLM, an open-source toolkit designed to simplify the implementation, understanding, and evaluation of LLM watermarking algorithms. It provides a unified framework, user-friendly interfaces, and visualization tools to enhance comprehension, along with a comprehensive suite of evaluation tools to support researchers and the broader community in advancing LLM watermarking technology.

### 45. [GPT-4 Jailbreaks Itself with Near-Perfect Success Using Self-Explanation](https://arxiv.org/pdf/2405.13077)

**Summary**: The paper introduces Iterative Refinement Induced Self-Jailbreak (IRIS), a novel method that uses the reflective capabilities of LLMs to jailbreak them with high success rates. By leveraging self-explanation and iterative refinement, IRIS achieves near-perfect jailbreak success rates on GPT-4, GPT-4 Turbo, and Llama-3.1-70B with minimal queries, outperforming previous approaches in efficiency and interpretability.

### 46. [How Do Humans Write Code? Large Models Do It the Same Way Too](https://arxiv.org/pdf/2402.15729)

**Summary**: The paper introduces Human-Think Language (HTL), a method that integrates Program-of-Thought (PoT) and Chain-of-Thought (CoT) to improve reasoning in Large Language Models (LLMs). HTL employs strategies like full CoT reasoning for code generation, Focus Attention to enhance logical code production, and reinforcement learning to optimize reasoning steps. This approach yields significant improvements in mathematical reasoning tasks, with an average increase of 6.5% and 4.3% on Llama-Base and Mistral-Base models, respectively, across multiple datasets, and demonstrates strong transferability to non-mathematical tasks.

### 47. [Open-Source Conversational AI with SpeechBrain 1.0](https://arxiv.org/pdf/2407.00463)

**Summary**: The paper introduces SpeechBrain 1.0, an open-source Conversational AI toolkit built on PyTorch, which now includes over 200 recipes and 100 models for various speech and language processing tasks. This update enhances support for diverse learning modalities, integrates Large Language Models, and introduces advanced decoding strategies, along with a new benchmark repository for unified model evaluation.

### 48. [PKU-SafeRLHF: Towards Multi-Level Safety Alignment for LLMs with Human Preference](https://arxiv.org/pdf/2406.15513)

**Summary**: The paper introduces PKU-SafeRLHF, a dataset designed to enhance safety alignment in LLMs by separating annotations of helpfulness and harmlessness. It provides 44.6k refined prompts and 265k question-answer pairs with safety meta-labels, covering 19 harm categories and three severity levels. The dataset also includes 166.8k preference data to train severity-sensitive moderation and safety-centric RLHF algorithms, aiming to support safer LLM deployment.

### 49. [Reward-Robust RLHF in LLMs](https://arxiv.org/pdf/2409.15360)

**Summary**: The paper introduces a reward-robust Reinforcement Learning from Human Feedback (RLHF) framework for Large Language Models (LLMs), addressing issues like reward hacking and misalignment by using Bayesian Reward Model Ensembles (BRME) to balance performance and robustness. The framework outperforms traditional methods in various benchmarks, offering improved accuracy and stability, and its theoretical analysis suggests it approaches the stability of constant reward settings.

### 50. [Recent advancements in LLM Red-Teaming: Techniques, Defenses, and Ethical Considerations](https://arxiv.org/pdf/2410.09097)

**Summary**: The paper reviews recent advancements in red-teaming Large Language Models (LLMs), focusing on attack strategies such as gradient-based optimization, reinforcement learning, and prompt engineering, and their impact on LLM security. It emphasizes the importance of developing robust defense mechanisms to counter these vulnerabilities, aiming to enhance the reliability and safety of LLMs.

### 51. [Knowledge-Augmented Reasoning for EUAIA Compliance and Adversarial Robustness of LLMs](https://arxiv.org/pdf/2410.09078)

**Summary**: The paper introduces a functional architecture to address the dual challenge of ensuring compliance with the EU AI Act (EUAIA) and achieving adversarial robustness in LLMs. By integrating a reasoning layer based on knowledge augmentation, the proposed system aims to support developers and auditors in verifying both compliance and robustness, thereby enhancing the trustworthiness of deployed LLMs in the EU.

### 52. [M3Hop-CoT: Misogynous Meme Identification with Multimodal Multi-hop Chain-of-Thought](https://arxiv.org/pdf/2410.09220)

**Summary**: The paper introduces M3Hop-CoT, a novel framework for identifying misogynous memes by integrating multimodal data and multi-hop Chain-of-Thought reasoning. The framework combines a CLIP-based classifier with a multimodal CoT module to enhance detection by considering cultural diversity, emotions, and contextual knowledge. Evaluations on the SemEval-2022 Task 5 dataset and other benchmarks demonstrate M3Hop-CoT's superior performance in macro-F1 score and generalizability across different datasets.

### 53. [Impeding LLM-assisted Cheating in Introductory Programming Assignments via Adversarial Perturbations](https://arxiv.org/pdf/2410.09318)

**Summary**: The paper explores the use of adversarial perturbations to hinder the performance of LLMs in generating code for introductory programming assignments, aiming to prevent cheating. A user study found that combined perturbations reduced the average correctness score by 77%, with the effectiveness of these perturbations influenced by their detectability.

### 54. [Sui Generis: Large Language Models for Authorship Attribution and Verification in Latin](https://arxiv.org/pdf/2410.09245)

**Summary**: The paper examines the effectiveness of Large Language Models (LLMs) in authorship attribution and verification for Latin texts from the Patristic Era, finding that LLMs can perform well in zero-shot scenarios but are susceptible to semantic confusion. The study highlights the challenges in guiding LLMs to make nuanced and explainable decisions, particularly when compared to their performance in high-resource modern languages.

### 55. [Nudging: Inference-time Alignment via Model Collaboration](https://arxiv.org/pdf/2410.09300)

**Summary**: The paper introduces "nudging," a training-free algorithm that aligns large language models at inference time using a smaller aligned model. By leveraging the base model's uncertainty in generating certain stylistic tokens, nudging effectively steers the output towards desired directions, achieving performance comparable to or better than fully aligned models without additional training. This method enables modular collaboration between different model families, offering a computationally efficient solution to model alignment.

### 56. [Keys to Robust Edits: from Theoretical Insights to Practical Advances](https://arxiv.org/pdf/2410.09338)

**Summary**: The paper investigates the limitations of current knowledge editing techniques in LLMs, particularly their lack of robustness in handling long contexts and paraphrased subjects. Through theoretical analysis, the authors identify key-value modeling issues and propose a novel 'group discussion' model to improve robustness. They introduce the Robust Edit Pathway (REP) to separate editing keys from LLM representations, demonstrating significant improvements in robustness across multiple metrics with minimal performance trade-offs.

### 57. [FB-Bench: A Fine-Grained Multi-Task Benchmark for Evaluating LLMs' Responsiveness to Human Feedback](https://arxiv.org/pdf/2410.09412)

**Summary**: The paper introduces FB-Bench, a comprehensive benchmark designed to evaluate the responsiveness of Large Language Models (LLMs) to human feedback in real-world scenarios. FB-Bench includes 734 curated samples across eight task types, five response deficiencies, and nine feedback types, revealing significant variations in LLM performance. The study highlights the importance of considering task complexity, feedback type, and response deficiencies in assessing LLM capabilities.

### 58. [Solving the Challenge Set without Solving the Task: On Winograd Schemas as a Test of Pronominal Coreference Resolution](https://arxiv.org/pdf/2410.09448)

**Summary**: The paper challenges the assumption that high performance on the Winograd Schema Challenge (WSC) necessarily indicates strong performance in resolving pronominal coreference in general. It demonstrates that while language models excel on WSC, they struggle with simpler coreference tasks in other datasets like OntoNotes. The authors propose an ensemble method combining language models with task-specific systems to improve overall coreference resolution accuracy and highlight the need for comprehensive evaluation across diverse datasets.

### 59. [CollabEdit: Towards Non-destructive Collaborative Knowledge Editing](https://arxiv.org/pdf/2410.09508)

**Summary**: The paper introduces CollabEdit, a non-destructive collaborative knowledge editing framework for LLMs, addressing challenges like knowledge overlap, conflict, and forgetting. By using a novel model merging mechanism, CollabEdit mimics global knowledge editing while preventing performance drops, outperforming destructive baselines in experiments on two datasets.

### 60. [Are You Human? An Adversarial Benchmark to Expose LLMs](https://arxiv.org/pdf/2410.09569)

**Summary**: The paper introduces an adversarial benchmark to detect Large Language Models (LLMs) in conversations by using implicit and explicit challenges. The benchmark, tested on nine leading models, shows that explicit challenges are more effective in identifying LLMs, while implicit challenges are less so. The study also highlights the prevalence of human participants using LLMs, emphasizing the need for reliable detection methods in critical interactions.

### 61. [Extended Japanese Commonsense Morality Dataset with Masked Token and Label Enhancement](https://arxiv.org/pdf/2410.09564)

**Summary**: The paper introduces the Extended Japanese Commonsense Morality (eJCM) dataset, which expands the original JCM dataset from 13,975 to 31,184 sentences using a novel method called Masked Token and Label Enhancement (MTLE). This method improves the dataset's cultural relevance and complexity, leading to a model trained on eJCM achieving higher F1 scores in moral reasoning tasks, particularly in culturally nuanced scenarios, compared to other models and augmentation techniques.

### 62. [Quebec Automobile Insurance Question-Answering With Retrieval-Augmented Generation](https://arxiv.org/pdf/2410.09623)

**Summary**: The paper investigates the use of Retrieval-Augmented Generation (RAG) with a state-of-the-art Large Language Model (LLM) to answer Quebec automobile insurance questions, leveraging a specialized corpus. It finds that while the RAG approach generally improves response quality, there is a significant risk of misinformation, with 5% to 13% of answers containing false statements that could mislead customers.

### 63. [Honest AI: Fine-Tuning "Small" Language Models to Say "I Don't Know", and Reducing Hallucination in RAG](https://arxiv.org/pdf/2410.09699)

**Summary**: The paper introduces Honest AI, a strategy to fine-tune smaller language models to reduce hallucination by teaching them to say "I don't know." It explores various approaches, including Retrieval-Augmented Generation (RAG) and fine-tuning, finding that a hybrid approach combining both methods performs best. The study highlights the effectiveness of smaller models, emphasizing resource efficiency.

### 64. [Taming Overconfidence in LLMs: Reward Calibration in RLHF](https://arxiv.org/pdf/2410.09724)

**Summary**: The paper addresses the issue of overconfidence in Large Language Models (LLMs) trained with Reinforcement Learning from Human Feedback (RLHF), revealing that RLHF tends to induce verbalized overconfidence in model responses. To mitigate this, the authors propose two PPO variants, PPO-M and PPO-C, which calibrate reward models and reward calculations, respectively, to better align confidence with response quality. Experimental results show that these methods reduce calibration error without compromising model performance.

### 65. ['Quis custodiet ipsos custodes?' Who will watch the watchmen? On Detecting AI-generated peer-reviews](https://arxiv.org/pdf/2410.09770)

**Summary**: The paper introduces two models, Term Frequency (TF) and Review Regeneration (RR), to detect AI-generated peer-reviews, addressing concerns about the integrity of the peer-review process in academic publishing. The TF model identifies repeated tokens by AI, while the RR model detects similar outputs from ChatGPT upon re-prompting. Both models are tested against token attacks and paraphrasing, with the RR model proving more robust. The study concludes that these methods outperform other AI text detectors and makes its resources publicly available.

### 66. [MisinfoEval: Generative AI in the Era of "Alternative Facts"](https://arxiv.org/pdf/2410.09949)

**Summary**: The paper introduces MisinfoEval, a framework for evaluating large language model (LLM) based interventions to combat misinformation on social media. Experiments in simulated environments show that LLM-based interventions significantly improve users' accuracy in identifying misinformation, with personalized explanations tailored to users' demographics and beliefs being particularly effective, increasing accuracy by up to 41.72%.

### 67. [When Neutral Summaries are not that Neutral: Quantifying Political Neutrality in LLM-Generated News Summaries](https://arxiv.org/pdf/2410.09978)

**Summary**: The study investigates the political neutrality of LLMs in generating summaries of polarizing news articles, focusing on five key US political issues. It finds a consistent pro-Democratic bias in several LLMs, particularly in gun control and healthcare, with significant vocabulary overlap in the generated summaries that align with Democratic perspectives. The findings are particularly relevant given the upcoming US elections.

### 68. [RMB: Comprehensively Benchmarking Reward Models in LLM Alignment](https://arxiv.org/pdf/2410.09893)

**Summary**: The paper introduces RMB, a comprehensive benchmark for evaluating reward models (RMs) in large language model (LLM) alignment, addressing limitations in current evaluation methods. RMB covers 49 real-world scenarios and employs both pairwise and Best-of-N evaluations to better reflect RM effectiveness. The study reveals generalization defects in state-of-the-art RMs and explores the potential of generative RMs, while also examining the impact of evaluation criteria and instructing methods.

### 69. [Evaluating Gender Bias of LLMs in Making Morality Judgements](https://arxiv.org/pdf/2410.09992)

**Summary**: The paper evaluates gender bias in LLMs when making moral judgments, using a new dataset called GenMO. It finds that despite safety measures, all tested models, including GPT-3.5-turbo and Llama 3, exhibit significant gender bias, often favoring female characters. The study also examines how model parameters influence bias and explores real-world scenarios where LLMs reveal biased moral decisions.

### 70. [Safety-Aware Fine-Tuning of Large Language Models](https://arxiv.org/pdf/2410.10014)

**Summary**: The paper introduces a Safety-Aware Fine-Tuning (SAFT) framework to automatically detect and remove harmful data during the fine-tuning of Large Language Models (LLMs), addressing the challenge of labor-intensive manual filtering. The framework leverages a scoring function based on subspace information to reduce harmfulness by up to 27.8%, demonstrating its effectiveness across various models and contamination rates, and validating its practical applicability in real-world scenarios.

### 71. [Diagnosing Hate Speech Classification: Where Do Humans and Machines Disagree, and Why?](https://arxiv.org/pdf/2410.10153)

**Summary**: The study investigates discrepancies between human and machine hate speech classification using cosine similarity, embedding regression, and manual re-annotation. It finds that female annotators are more sensitive to racial slurs, and while machines achieve high accuracy, they struggle with short swear words due to model alignment constraints.

### 72. [Jailbreak Instruction-Tuned LLMs via end-of-sentence MLP Re-weighting](https://arxiv.org/pdf/2410.10150)

**Summary**: The paper explores vulnerabilities in instruction-tuned LLMs by re-weighting Multi-Layer Perceptron (MLP) neurons, particularly in end-of-sentence inferences. The authors develop two white-box jailbreak methods—prompt-specific and prompt-general—to compromise model safety, demonstrating effectiveness across various LLM sizes. This study highlights the critical role of MLP layers in safety evaluation and offers insights into LLM vulnerabilities.

### 73. [Minimum Tuning to Unlock Long Output from LLMs with High Quality Data as the Key](https://arxiv.org/pdf/2410.10210)

**Summary**: The paper investigates the impact of high-quality data on tuning LLMs to generate longer outputs, suggesting that careful data curation can significantly enhance long-output capabilities with minimal computational resources. The authors demonstrate consistent improvements across various models by using a small, high-quality dataset, and have made their dataset, tuning methods, and fine-tuned models publicly available.

### 74. [Locking Down the Finetuned LLMs Safety](https://arxiv.org/pdf/2410.10343)

**Summary**: The paper introduces SafetyLock, a method to maintain the safety of fine-tuned LLMs by leveraging the similarity of safety-related activation patterns between fine-tuned and base models. SafetyLock extracts Meta-SafetyLock, a set of safety bias directions, which are then applied to fine-tuned models to enhance their safety without additional computational cost. The method significantly reduces harmful responses and outperforms traditional safety measures in both performance and efficiency.

### 75. [Parenting: Optimizing Knowledge Selection of Retrieval-Augmented Language Models with Parameter Decoupling and Tailored Tuning](https://arxiv.org/pdf/2410.10360)

**Summary**: The paper introduces Parenting, a framework that optimizes knowledge selection in Retrieval-Augmented Generation (RAG) by decoupling adherence and robustness within the parameter space of Large Language Models (LLMs). By identifying crucial parameter units through forward activation gain and applying type-guided tailored tuning, Parenting aims to balance model performance and robustness, demonstrated through extensive experiments across different datasets and models.

### 76. [Medico: Towards Hallucination Detection and Correction with Multi-source Evidence Fusion](https://arxiv.org/pdf/2410.10408)

**Summary**: The paper introduces Medico, a framework for detecting and correcting hallucinations in Large Language Models by fusing evidence from multiple sources. Medico not only identifies factual errors in generated content but also provides rationales for its judgments and iteratively revises the erroneous information. The framework demonstrates strong performance in evidence retrieval, hallucination detection, and correction, with high approval rates in experimental results.

### 77. [Generative AI and Its Impact on Personalized Intelligent Tutoring Systems](https://arxiv.org/pdf/2410.10650)

**Summary**: The paper explores how Generative AI, particularly large language models like GPT-4, can enhance Intelligent Tutoring Systems (ITS) by providing personalized, adaptive learning experiences through dynamic content generation and real-time feedback. It discusses applications such as automated question creation and interactive dialogue systems, while also addressing challenges like pedagogical accuracy and AI biases. The report emphasizes the transformative potential of Generative AI in making education more effective and equitable.

### 78. [Derail Yourself: Multi-turn LLM Jailbreak Attack through Self-discovered Clues](https://arxiv.org/pdf/2410.10700)

**Summary**: The paper introduces ActorAttack, a multi-turn attack method that exploits the vulnerabilities of Large Language Models (LLMs) by obscuring harmful intents across multiple queries. Inspired by actor-network theory, ActorAttack models a network of semantically linked actors to generate diverse attack paths, effectively concealing harmful intents and uncovering multiple attack routes. The study demonstrates that ActorAttack outperforms existing attack methods and enhances the robustness of safety-tuned models using the SafeMTData dataset.

### 79. [Mix Data or Merge Models? Optimizing for Diverse Multi-Task Learning](https://arxiv.org/pdf/2410.10801)

**Summary**: The paper investigates the effectiveness of model merging versus data mixing for diverse multi-task learning in multilingual settings, particularly focusing on improving both general performance and safety. The study finds that objective-based merging outperforms data mixing, with significant improvements in both areas, and that language-based merging of monolingually fine-tuned models further enhances performance and reduces harm.

### 80. [When Attention Sink Emerges in Language Models: An Empirical View](https://arxiv.org/pdf/2410.10781)

**Summary**: The paper investigates the phenomenon of attention sink in Language Models (LMs), where significant attention is disproportionately assigned to the first token. The study reveals that attention sinks are prevalent across various LMs and emerge during pre-training, influenced by optimization, data distribution, and loss functions. The authors suggest that attention sinks act as key biases and can be mitigated by altering the attention mechanism, such as using sigmoid attention without normalization.

### 81. [Local and Global Decoding in Text Generation](https://arxiv.org/pdf/2410.10810)

**Summary**: The paper explores the impact of local versus global decoding methods in text generation, particularly in dialogue systems, by comparing top-$k$ and top-$\pi$ algorithms with their globally-normalized counterparts. It introduces an independent Metropolis-Hastings algorithm to approximate global sampling. Empirical results indicate that global decoding generally performs worse than local decoding, suggesting that the distortion introduced by local methods is a beneficial feature in text generation.

### 82. [Can a large language model be a gaslighter?](https://arxiv.org/pdf/2410.09181)

**Summary**: The paper investigates the potential for LLMs to engage in gaslighting, a psychological manipulation, through prompt-based and fine-tuning-based attacks. The authors propose a two-stage framework, DeepCoG, to elicit gaslighting behaviors in LLMs and demonstrate that such attacks can effectively turn LLMs into gaslighters. They also introduce safety alignment strategies to enhance LLMs' resistance to gaslighting without significantly compromising their utility.

### 83. [VLFeedback: A Large-Scale AI Feedback Dataset for Large Vision-Language Models Alignment](https://arxiv.org/pdf/2410.09421)

**Summary**: The paper introduces VLFeedback, a large-scale dataset of over 82K multi-modal instructions and rationales generated by AI models to align large vision-language models (LVLMs) without human annotations. The authors demonstrate the effectiveness of AI feedback by fine-tuning an LVLM named Silkie, which shows significant improvements in helpfulness, visual faithfulness, and safety, outperforming the base model in various tasks and reducing hallucination issues.

### 84. [Survival of the Safest: Towards Secure Prompt Optimization through Interleaved Multi-Objective Evolution](https://arxiv.org/pdf/2410.09652)

**Summary**: The paper introduces "Survival of the Safest" (SoS), a multi-objective prompt optimization framework for LLMs that simultaneously enhances performance and security. SoS employs an interleaved multi-objective evolution strategy to efficiently optimize prompts in high-dimensional spaces, outperforming traditional single-objective methods in both performance and security across various benchmarks.

### 85. [Collu-Bench: A Benchmark for Predicting Language Model Hallucinations in Code](https://arxiv.org/pdf/2410.09997)

**Summary**: The paper introduces Collu-Bench, a benchmark for evaluating and predicting hallucinations in code generated by LLMs, focusing on code generation and automated program repair tasks. The benchmark includes 13,234 instances from various LLMs and datasets, providing detailed features for analysis. Experiments show that predicting code hallucinations remains challenging, with accuracy ranging from 22.03% to 33.15%, underscoring the need for more advanced techniques.

### 86. [Denial-of-Service Poisoning Attacks against Large Language Models](https://arxiv.org/pdf/2410.10760)

**Summary**: The paper introduces Denial-of-Service Poisoning (P-DoS) attacks against Large Language Models (LLMs), demonstrating how a single poisoned sample can bypass output length limits and cause excessive repetition. The study highlights the vulnerability of LLMs to such attacks and emphasizes the need for robust defenses to protect against them.

### 87. [On Calibration of LLM-based Guard Models for Reliable Content Moderation](https://arxiv.org/pdf/2410.10414)

**Summary**: The paper investigates the reliability and calibration of existing LLM-based guard models used for content moderation, finding that these models often produce overconfident predictions and are vulnerable to jailbreak attacks. The study introduces post-hoc calibration methods, particularly temperature scaling and contextual calibration, to improve model reliability. The findings highlight the need for rigorous calibration evaluation in future guard model development.

### 88. [Untying the Reversal Curse via Bidirectional Language Model Editing](https://arxiv.org/pdf/2310.10322)

**Summary**: The paper introduces a new approach to evaluating and improving the bidirectional recall of edited knowledge in LLMs. It identifies a "reversal curse" where models struggle to recall edited facts in the reverse direction and proposes a method called Bidirectionally Inversible Relationship moDeling (BIRD) to address this issue. BIRD enhances model performance by incorporating bidirectional relationships into the editing process, as demonstrated through experiments on various LLMs.

### 89. [LLM Task Interference: An Initial Study on the Impact of Task-Switch in Conversational History](https://arxiv.org/pdf/2402.18216)

**Summary**: The paper investigates the impact of task-switching in conversational history on the performance of LLMs. It finds that task-switches can lead to significant performance degradation, and the study formalizes this vulnerability for the first time, highlighting the need for better context management in conversational AI systems.

### 90. [Extreme Miscalibration and the Illusion of Adversarial Robustness](https://arxiv.org/pdf/2402.17509)

**Summary**: The paper reveals that miscalibration in NLP models can create an illusion of robustness against adversarial attacks, as it masks gradients and hinders attack methods. The authors demonstrate that test-time temperature calibration can expose this illusion, and advocate for its inclusion in robustness evaluations to ensure genuine model resilience. They also propose training-time temperature scaling as a method to enhance true robustness.

### 91. [Likelihood-based Mitigation of Evaluation Bias in Large Language Models](https://arxiv.org/pdf/2402.15987)

**Summary**: The paper investigates the likelihood bias in Large Language Models (LLMs) used for evaluating natural language generation tasks, finding that LLMs tend to favor sentences with higher likelihoods. To address this, the authors propose a method using highly biased instances for in-context learning, which effectively mitigates the bias and significantly improves the correlation between model evaluations and human scores.

### 92. [Spotting LLMs With Binoculars: Zero-Shot Detection of Machine-Generated Text](https://arxiv.org/pdf/2401.12070)

**Summary**: The paper introduces Binoculars, a novel zero-shot method for detecting machine-generated text using a pair of pre-trained LLMs. By contrasting the outputs of these models, Binoculars achieves state-of-the-art accuracy in identifying machine-generated text across various document types, with over 90% detection rate for models like ChatGPT at a low false positive rate, without requiring any training data specific to the target LLM.

### 93. [Should We Respect LLMs? A Cross-Lingual Study on the Influence of Prompt Politeness on LLM Performance](https://arxiv.org/pdf/2402.14531)

**Summary**: The study examines how politeness in prompts affects the performance of LLMs across English, Chinese, and Japanese tasks. It finds that impolite prompts generally lead to poorer performance, but overly polite language does not necessarily improve outcomes. The optimal level of politeness varies by language, indicating that LLMs are influenced by cultural norms and language-specific contexts, emphasizing the importance of considering politeness in cross-cultural NLP and LLM applications.

### 94. [Don't Say No: Jailbreaking LLM by Suppressing Refusal](https://arxiv.org/pdf/2404.16369)

**Summary**: The paper introduces the DSN (Don't Say No) attack, which enhances jailbreaking of Large Language Models (LLMs) by suppressing their refusal to generate harmful content. It also proposes an Ensemble Evaluation pipeline that combines Natural Language Inference (NLI) and external LLM evaluators to more accurately assess the harmfulness of responses, outperforming existing methods.

### 95. [NoiseBench: Benchmarking the Impact of Real Label Noise on Named Entity Recognition](https://arxiv.org/pdf/2405.07609)

**Summary**: The paper introduces NoiseBench, a benchmark for evaluating the impact of real label noise on Named Entity Recognition (NER). Unlike previous studies that use simulated noise, NoiseBench incorporates six types of real noise, revealing that real noise is significantly more challenging for models. The study shows that current noise-robust learning models perform poorly under real noise conditions, highlighting the need for more effective approaches.

### 96. [Self-Control of LLM Behaviors by Compressing Suffix Gradient into Prefix Controller](https://arxiv.org/pdf/2406.02721)

**Summary**: The paper introduces SelfControl, an inference-time method for controlling LLMs by using gradients derived from a natural language suffix to guide the model's behavior. This approach eliminates the need for human annotations and provides precise, transparent, and adaptable control. The authors also propose SelfControl_{Prefix}, a compact module that enhances efficiency by encapsulating learned representations, enabling simultaneous control of multiple behaviors without added latency.

### 97. [Know the Unknown: An Uncertainty-Sensitive Method for LLM Instruction Tuning](https://arxiv.org/pdf/2406.10099)

**Summary**: The paper introduces an uncertainty-sensitive tuning method for LLMs to improve their ability to recognize and admit when they lack knowledge, reducing hallucinations. The two-stage training approach, involving uncertainty recognition and prompt-sensitive activation, significantly enhances the model's performance in handling questions with knowledge gaps, outperforming GPT-4 in some cases.

### 98. [Buffer of Thoughts: Thought-Augmented Reasoning with Large Language Models](https://arxiv.org/pdf/2406.04271)

**Summary**: The paper introduces Buffer of Thoughts (BoT), a method that enhances LLMs by storing and adapting high-level thought templates from various tasks to improve reasoning efficiency and accuracy. BoT achieves significant performance improvements across multiple reasoning tasks, demonstrating superior generalization and robustness while reducing computational costs compared to existing methods.

### 99. [Investigating Annotator Bias in Large Language Models for Hate Speech Detection](https://arxiv.org/pdf/2406.11109)

**Summary**: The paper investigates biases in Large Language Models (LLMs) like GPT-3.5 and GPT-4 when used for hate speech detection, focusing on gender, race, religion, and disability. It introduces the HateSpeechCorpus dataset and compares results with the ETHOS dataset to understand and mitigate annotator biases in LLMs, aiming to improve the reliability of hate speech detection systems.

### 100. [Designing a Dashboard for Transparency and Control of Conversational AI](https://arxiv.org/pdf/2406.07882)

**Summary**: The paper introduces a dashboard designed to enhance transparency and control in conversational AI systems by revealing the internal user model of a large language model (LLM). The dashboard allows users to see and adjust aspects like age, gender, and socioeconomic status, which influences the AI's responses. A study found that users felt more in control and could better identify biases, suggesting the dashboard's potential to improve AI interactions.



---

*Last updated on 2024-10-17*