# Awesome-LLM-Jailbreak

## Awesome LLM Jailbreak

Welcome to the **Awesome LLM Jailbreak** repository! This project curates a list of high-quality resources related to LLM Jailbreak, including research papers, tools, libraries, and more.

## Daily Updates

### 2024-10-14

### 1. [Understanding the Interplay between Parametric and Contextual Knowledge for Large Language Models](https://arxiv.org/pdf/2410.08414)

**Summary**: The paper explores how LLMs integrate parametric knowledge (PK) with contextual knowledge (CK) and identifies four types of interactions between them. Through the ECHOQA benchmark, it finds that LLMs often suppress PK in favor of CK, even when PK is more relevant, highlighting a vulnerability in their reliability for knowledge-intensive tasks.

### 2. [GUS-Net: Social Bias Classification in Text with Generalizations, Unfairness, and Stereotypes](https://arxiv.org/pdf/2410.08388)

**Summary**: The paper introduces GUS-Net, a novel approach for detecting three types of biases—generalizations, unfairness, and stereotypes—in text using generative AI and automated agents to create a synthetic dataset. GUS-Net outperforms existing methods by incorporating contextual encodings from pre-trained models, achieving higher accuracy and better bias identification across diverse contexts.

### 3. [Do You Know What You Are Talking About? Characterizing Query-Knowledge Relevance For Reliable Retrieval Augmented Generation](https://arxiv.org/pdf/2410.08320)

**Summary**: The paper introduces a statistical framework to assess the relevance of user queries to an external knowledge corpus in Retrieval Augmented Generation (RAG) systems, aiming to improve the reliability of generated responses. It proposes both online and offline testing methods to detect queries that are outside the scope of the knowledge corpus or are based on outdated information, thereby enhancing the overall quality of RAG systems.

### 4. [SocialGaze: Improving the Integration of Human Social Norms in Large Language Models](https://arxiv.org/pdf/2410.08698)

**Summary**: The paper introduces SocialGaze, a multi-step prompting framework designed to improve the alignment of LLMs with human social norms and values, particularly in judging social acceptability. By prompting models to consider social situations from multiple perspectives before forming a judgment, SocialGaze enhances the model's alignment with human consensus, as evidenced by a significant improvement in F1 scores. The study also highlights biases in LLMs, such as unfair judgments towards males and differences in alignment based on the age of narrators.

### 5. [On the State of NLP Approaches to Modeling Depression in Social Media: A Post-COVID-19 Outlook](https://arxiv.org/pdf/2410.08793)

**Summary**: The paper surveys NLP approaches to modeling depression in social media, focusing on the post-COVID-19 era, where the pandemic has significantly increased depression rates. It reviews state-of-the-art methods and new datasets used in this context, while also addressing ethical concerns related to data collection and processing in mental health research.

### 6. [Measuring the Groundedness of Legal Question-Answering Systems](https://arxiv.org/pdf/2410.08764)

**Summary**: The paper introduces a benchmark for assessing the groundedness of AI-generated responses in legal question-answering systems, using similarity-based metrics and natural language inference models. It explores different prompting strategies and validates the methods with a specialized grounding classification corpus, achieving a macro-F1 score of 0.8. The study highlights the potential of these methods to enhance the reliability and trustworthiness of generative AI in legal contexts.

### 7. [Measuring the Inconsistency of Large Language Models in Preferential Ranking](https://arxiv.org/pdf/2410.08851)

**Summary**: The paper examines the consistency of LLMs in generating preferential rankings, introducing a formalization based on order theory to assess criteria like transitivity and independence from irrelevant alternatives. The study finds that current LLMs often fail to meet these criteria, showing significant inconsistencies and biases, which suggests a need for further research to improve their reliability in decision-making scenarios.

### 8. [Which Demographics do LLMs Default to During Annotation?](https://arxiv.org/pdf/2410.08820)

**Summary**: The paper investigates how LLMs default to certain demographics during text annotation when no demographic information is provided. By comparing non-demographic, placebo-conditioned, and demographic-conditioned prompts, the study finds significant influences of gender, race, and age in demographic prompting, challenging previous findings of no such effects.

### 9. [NoVo: Norm Voting off Hallucinations with Attention Heads in Large Language Models](https://arxiv.org/pdf/2410.08970)

**Summary**: The paper introduces Norm Voting (NoVo), a lightweight method that leverages attention head norms in Large Language Models to significantly improve factual accuracy in zero-shot multiple-choice questions. NoVo achieves state-of-the-art performance on TruthfulQA MC1 and demonstrates exceptional generalization across diverse datasets, outperforming existing methods by a substantial margin.

### 10. [Controllable Safety Alignment: Inference-Time Adaptation to Diverse Safety Requirements](https://arxiv.org/pdf/2410.08968)

**Summary**: The paper introduces Controllable Safety Alignment (CoSA), a framework that allows LLMs to adapt to diverse safety requirements without retraining, by aligning models to safety configurations provided in natural language prompts. CoSAlign, a data-centric method, enables this adaptation, and the authors propose a new evaluation protocol, CoSA-Score, to measure both helpfulness and configured safety, supported by the CoSApien benchmark.

### 11. [Hypothesis-only Biases in Large Language Model-Elicited Natural Language Inference](https://arxiv.org/pdf/2410.08996)

**Summary**: The study investigates whether LLMs introduce annotation artifacts when generating hypotheses for Natural Language Inference (NLI) tasks. By training hypothesis-only classifiers on datasets created with GPT-4, Llama-2, and Mistral 7b, the researchers found high accuracy rates (86-96%), indicating the presence of hypothesis-only biases. The analysis also revealed frequent "give-aways" in LLM-generated hypotheses, suggesting that NLI biases can persist in LLM-generated data.

### 12. [AttnGCG: Enhancing Jailbreaking Attacks on LLMs with Attention Manipulation](https://arxiv.org/pdf/2410.09040)

**Summary**: The paper introduces AttnGCG, a method that enhances jailbreaking attacks on Large Language Models (LLMs) by manipulating their attention scores. This approach shows significant improvements in attack efficacy, with an average increase of 7% for Llama-2 and 10% for Gemma series models, and demonstrates robust transferability against unseen goals and black-box models like GPT-3.5 and GPT-4.

### 13. [Unraveling and Mitigating Safety Alignment Degradation of Vision-Language Models](https://arxiv.org/pdf/2410.09047)

**Summary**: The paper investigates the degradation of safety alignment in Vision-Language Models (VLMs) due to the integration of vision modules, which creates a representation gap between text-only and multi-modal inputs. To address this, the authors propose Cross-Modality Representation Manipulation (CMRM), an inference-time intervention method that recovers the safety alignment capabilities of the underlying Language Model backbone without additional training, significantly reducing the unsafe rate in multi-modal inputs.

### 14. ["I Am the One and Only, Your Cyber BFF": Understanding the Impact of GenAI Requires Understanding the Impact of Anthropomorphic AI](https://arxiv.org/pdf/2410.08526)

**Summary**: The paper argues that the increasing anthropomorphism in generative AI systems, where outputs are perceived as human-like, has significant social impacts that are understudied. The authors emphasize the need to understand these anthropomorphic aspects to fully grasp the societal implications of generative AI, and call for more research in this area.

### 15. [Simultaneous Reward Distillation and Preference Learning: Get You a Language Model Who Can Do Both](https://arxiv.org/pdf/2410.08458)

**Summary**: The paper introduces DRDO (Direct Reward Distillation and policy-Optimization), a method that combines reward modeling and preference learning to improve the performance of generative large language models. DRDO avoids issues like model drift and reward overfitting by directly mimicking rewards and learning preferences, outperforming previous methods like DPO and e-DPO in terms of expected rewards and robustness to noisy data and out-of-distribution settings.

### 16. [HyperDPO: Hypernetwork-based Multi-Objective Fine-Tuning Framework](https://arxiv.org/pdf/2410.08316)

**Summary**: The paper introduces HyperDPO, a hypernetwork-based framework for Multi-Objective Fine-Tuning (MOFT) that extends Direct Preference Optimization (DPO) to handle listwise ranking datasets using the Plackett-Luce model. HyperDPO enables efficient one-shot training for profiling the Pareto front and offers flexible post-training control over trade-offs, with a novel Hyper Prompt Tuning design for continuous weight adjustment across objectives. The framework is shown to be effective and efficient in various tasks, including Learning-to-Rank and LLM alignment.

### 17. [PoisonBench: Assessing Large Language Model Vulnerability to Data Poisoning](https://arxiv.org/pdf/2410.08811)

**Summary**: The paper introduces PoisonBench, a benchmark for evaluating large language models' vulnerability to data poisoning during preference learning. It finds that model size does not guarantee resilience, there is a log-linear relationship between attack effects and poison ratio, and poisoning effects can generalize to new triggers. These findings underscore the need for more robust defenses against data poisoning attacks.

### 18. [AgentHarm: A Benchmark for Measuring Harmfulness of LLM Agents](https://arxiv.org/pdf/2410.09024)

**Summary**: The paper introduces AgentHarm, a benchmark designed to assess the harmfulness and robustness of LLM agents against malicious tasks, focusing on scenarios where agents use external tools and execute multi-stage tasks. The benchmark includes 110 malicious tasks across 11 harm categories and evaluates the ability of models to resist and recover from jailbreak attacks. The study finds that leading LLMs are often compliant with harmful requests and that simple jailbreak methods can effectively compromise agent capabilities, highlighting the need for improved defenses.

### 19. [MiRAGeNews: Multimodal Realistic AI-Generated News Detection](https://arxiv.org/pdf/2410.09045)

**Summary**: The paper introduces MiRAGeNews, a dataset of 12,500 real and AI-generated image-caption pairs designed to challenge both human and AI detection of fake news. The authors develop a multi-modal detector, MiRAGe, which outperforms existing models by improving F-1 scores by 5.1% on out-of-domain data. The dataset and code are made publicly available to support research in AI-generated content detection.

### 20. [NatLogAttack: A Framework for Attacking Natural Language Inference Models with Natural Logic](https://arxiv.org/pdf/2307.02849)

**Summary**: The paper introduces NatLogAttack, a framework for attacking natural language inference (NLI) models using natural logic, a classical logic formalism. The framework generates both label-preserving and label-flipping adversarial examples, demonstrating that NLI models are more vulnerable under label-flipping attacks. The study highlights the potential of logic-based attacks to evaluate and improve the reasoning capabilities of NLI models.

### 21. [Do Large Language Models have Shared Weaknesses in Medical Question Answering?](https://arxiv.org/pdf/2310.07225)

**Summary**: The study benchmarks 16 LLMs on 874 Polish medical licensing exam questions to identify shared weaknesses and strengths. Results show that LLM accuracies are positively correlated, with performance linked to human test taker scores and negatively correlated with question difficulty. Larger models generally perform better, but training, architecture, and data differences also significantly impact accuracy.

### 22. [CMD: a framework for Context-aware Model self-Detoxification](https://arxiv.org/pdf/2308.08295)

**Summary**: The paper introduces the Context-aware Model self-Detoxification (CMD) framework, which addresses the challenge of balancing detoxification effectiveness and generation quality by first detoxifying the context and then guiding the language model to generate content aligned with the safe context. The CMD framework includes a two-phase process and a toxic contrastive loss to enhance detoxification while maintaining semantic coherence, demonstrating superior performance across various language models compared to existing methods.

### 23. [Unintentional Unalignment: Likelihood Displacement in Direct Preference Optimization](https://arxiv.org/pdf/2410.08847)

**Summary**: The paper investigates "likelihood displacement," a phenomenon in Direct Preference Optimization (DPO) where the likelihood of preferred responses decreases during training, potentially leading to unintentional unalignment. The authors demonstrate that this displacement can shift probability mass to responses with opposite meanings and show its impact on refusing unsafe prompts. They introduce the Centered Hidden Embedding Similarity (CHES) score to identify and mitigate this issue by filtering training samples with similar embeddings, emphasizing the need for distinct preferences in data curation.

### 24. [Controllable Preference Optimization: Toward Controllable Multi-Objective Alignment](https://arxiv.org/pdf/2402.19085)

**Summary**: The paper introduces Controllable Preference Optimization (CPO) to address the "alignment tax" in AI models, where improving one aspect of alignment can negatively impact others. CPO allows for explicit preference specification across multiple objectives, guiding models to balance "helpfulness, honesty, and harmlessness" effectively. Experimental results show that CPO outperforms traditional methods in achieving multi-objective alignment while mitigating the alignment tax.

### 25. [Influence of Solution Efficiency and Valence of Instruction on Additive and Subtractive Solution Strategies in Humans and GPT-4](https://arxiv.org/pdf/2404.16692)

**Summary**: The study compares human and GPT-4 problem-solving strategies in spatial and linguistic tasks, finding that GPT-4 exhibits a stronger addition bias than humans, especially when subtraction is more efficient. Additionally, GPT-4's use of additive strategies increases with positive valence instructions, highlighting differences in efficiency-based strategies between humans and LLMs, and underscoring the need for caution in their real-world applications.

### 26. [LLM-Generated Black-box Explanations Can Be Adversarially Helpful](https://arxiv.org/pdf/2405.06800)

**Summary**: The paper highlights a risk in using Large Language Models (LLMs) for black-box explanations, termed *adversarial helpfulness*, where LLMs can make incorrect answers appear correct through persuasive strategies. The study reveals that these models use tactics like reframing questions and expressing high confidence to mislead, and that they struggle with navigating complex knowledge structures when generating such explanations. This underscores the limitations of LLMs in black-box settings and offers guidance for their safer use.

### 27. [Controlling Large Language Model Agents with Entropic Activation Steering](https://arxiv.org/pdf/2406.00244)

**Summary**: The paper introduces Entropic Activation Steering (EAST), a method to control the exploration behavior of large language model (LLM) agents by manipulating high-level actions derived from the model's outputs. EAST modulates the uncertainty in the LLM's decision-making process, guiding the agent towards more exploratory actions, and demonstrates that the steering vectors generalize across different tasks. This approach offers a new perspective on understanding and controlling LLM agents' decision-making behaviors.

### 28. [Evaluating Copyright Takedown Methods for Language Models](https://arxiv.org/pdf/2406.18664)

**Summary**: The paper introduces CoTaEval, an evaluation framework to assess the effectiveness of copyright takedown methods for language models, focusing on their impact on model utility, efficiency, and retention of factual knowledge. The study finds that no single method performs optimally across all metrics, highlighting the need for further research in this area and suggesting challenges for current policy proposals.

### 29. [Judging the Judges: Evaluating Alignment and Vulnerabilities in LLMs-as-Judges](https://arxiv.org/pdf/2406.12624)

**Summary**: The paper evaluates the performance of various LLMs acting as judges in assessing other LLMs, finding that only the largest models achieve reasonable alignment with human judgments but still fall short of inter-human agreement. The study highlights vulnerabilities in judge models, such as sensitivity to prompt complexity and leniency, and emphasizes the need for caution in using LLMs as judges, especially in more complex scenarios.

### 30. [Modular Pluralism: Pluralistic Alignment via Multi-LLM Collaboration](https://arxiv.org/pdf/2406.15951)

**Summary**: The paper introduces Modular Pluralism, a framework that enhances LLMs by integrating specialized community LMs to better represent diverse human preferences. This modular approach supports three modes of pluralism and is compatible with both black-box and open-source LLMs, demonstrating improved performance in handling value-laden and perspective-informed tasks. The framework allows for easy expansion to include underrepresented communities by adding new community LMs.

### 31. [Cross-modality Information Check for Detecting Jailbreaking in Multimodal Large Language Models](https://arxiv.org/pdf/2407.21659)

**Summary**: The paper introduces CIDER, a plug-and-play jailbreaking detector for Multimodal Large Language Models (MLLMs), designed to identify maliciously perturbed image inputs by leveraging cross-modal similarity. CIDER enhances security without modifying the model's internal structure or requiring significant computational resources, demonstrating effectiveness and transferability across different MLLMs.

### 32. [Measuring and Enhancing Trustworthiness of LLMs in RAG through Grounded Attributions and Learning to Refuse](https://arxiv.org/pdf/2409.11242)

**Summary**: The paper introduces Trust-Score, a metric to evaluate the trustworthiness of LLMs in retrieval-augmented generation (RAG) systems. It finds that existing prompting methods do not effectively adapt LLMs for RAG tasks, leading to the development of Trust-Align, a method that significantly improves performance on benchmarks like ASQA, QAMPARI, and ELI5 across various LLM sizes.

### 33. [Teaching LLMs to Abstain across Languages via Multilingual Feedback](https://arxiv.org/pdf/2406.15948)

**Summary**: The paper introduces a novel approach to improve multilingual Large Language Models (LLMs) by teaching them to abstain from answering when facing knowledge gaps, particularly in under-resourced languages. The proposed method involves LLMs generating feedback in related languages to self-assess their responses, which helps identify knowledge gaps across diverse languages and cultures. This approach significantly enhances performance, especially for low-resource languages, and highlights the importance of cultural factors in language modeling.

### 34. [Wait, that's not an option: LLMs Robustness with Incorrect Multiple-Choice Options](https://arxiv.org/pdf/2409.00113)

**Summary**: The study investigates how LLMs respond to misleading instructions, finding that models like GPT-4 and Claude 3 tend to follow instructions without questioning the validity of options, while models like Llama 3.1 and Qwen2.5 show improved refusal rates with increased model size. The research introduces a "reflective judgment" metric to assess this behavior and suggests that alignment techniques can sometimes impair models' ability to reject incorrect instructions.

### 35. [Negative Preference Optimization: From Catastrophic Collapse to Effective Unlearning](https://arxiv.org/pdf/2404.05868)

**Summary**: The paper introduces Negative Preference Optimization (NPO), a novel method for unlearning undesirable data in Large Language Models (LLMs), which addresses the issues of catastrophic collapse and ineffective unlearning seen in gradient ascent-based approaches. NPO demonstrates superior performance in both synthetic and real-world datasets, achieving significant unlearning while preserving model utility, and is shown to be exponentially more stable than gradient ascent methods.

### 36. [Learn Your Reference Model for Real Good Alignment](https://arxiv.org/pdf/2404.09656)

**Summary**: The paper introduces Trust Region methods (TR-DPO, TR-IPO, TR-KTO) for aligning Large Language Models (LLMs) offline, addressing the issue of overoptimization by dynamically updating the reference policy during training. These methods effectively mitigate overoptimization, maintaining strong performance even with substantial deviations from the initial reference policy. The approaches are shown to outperform conventional methods in tasks like dialogue and summarization, and significantly improve general-purpose assistant performance on benchmarks like AlpacaEval 2 and Arena-Hard.

### 37. [PostMark: A Robust Blackbox Watermark for Large Language Models](https://arxiv.org/pdf/2406.14517)

**Summary**: The paper introduces PostMark, a robust blackbox watermarking method for LLMs that does not require access to the model's logits, addressing concerns about model distillation. PostMark inserts an input-dependent set of words into the text after decoding, making it more resistant to paraphrasing attacks compared to existing methods. The study evaluates PostMark across multiple LLMs and datasets, demonstrating its effectiveness while noting the trade-off between text quality and watermark robustness.

### 38. [Controlling Whisper: Universal Acoustic Adversarial Attacks to Control Speech Foundation Models](https://arxiv.org/pdf/2407.04482)

**Summary**: The paper explores the vulnerability of speech-enabled foundation models, such as OpenAI's Whisper, to adversarial attacks that can manipulate their behavior without altering the model's prompt. By prepending a universal adversarial acoustic segment to any input speech signal, the authors demonstrate the ability to override the model's intended task, such as forcing speech translation instead of transcription. This highlights a significant security concern for multi-tasking speech models that must be addressed before deployment.

### 39. [Unraveling Cross-Modality Knowledge Conflicts in Large Vision-Language Models](https://arxiv.org/pdf/2410.03659)

**Summary**: The paper addresses the issue of cross-modality parametric knowledge conflicts in Large Vision-Language Models (LVLMs), where inconsistencies between visual and textual components lead to errors. The authors introduce a systematic approach to detect and mitigate these conflicts, proposing a dynamic contrastive decoding method and prompt-based strategies to improve model accuracy. Their methods show significant improvements, particularly in accuracy on the ViQuAE and InfoSeek datasets.

### 40. [Localizing Factual Inconsistencies in Attributable Text Generation](https://arxiv.org/pdf/2410.07473)

**Summary**: The paper introduces QASemConsistency, a method for precisely localizing factual inconsistencies in generated text by decomposing it into minimal predicate-argument propositions expressed as QA pairs and comparing them to a trusted reference. The approach is shown to be effective for both human annotation and automated detection using supervised models and large language models, achieving high inter-annotator agreement.

### 41. [AI-Press: A Multi-Agent News Generating and Feedback Simulation System Powered by Large Language Models](https://arxiv.org/pdf/2410.07561)

**Summary**: The paper introduces AI-Press, a multi-agent system leveraging large language models for automated news generation and refinement, addressing limitations in professionalism and ethical judgment. It also includes a feedback simulation system to predict public reactions based on demographic data, demonstrating enhanced news-generating capabilities and the effectiveness of feedback prediction through evaluations.

### 42. [StablePrompt: Automatic Prompt Tuning using Reinforcement Learning for Large Language Models](https://arxiv.org/pdf/2410.07652)

**Summary**: The paper introduces StablePrompt, a method for automatic prompt tuning in Large Language Models (LLMs) using Reinforcement Learning (RL). It addresses the instability of RL in prompt tuning by employing Adaptive Proximal Policy Optimization (APPO), which uses an LLM anchor model to adaptively adjust policy updates, ensuring stable and high-performance prompt generation across various tasks.

### 43. [Detecting Training Data of Large Language Models via Expectation Maximization](https://arxiv.org/pdf/2410.07582)

**Summary**: The paper introduces EM-MIA, a novel membership inference attack (MIA) method for LLMs that uses an expectation-maximization algorithm to iteratively refine membership and prefix scores, achieving state-of-the-art results on the WikiMIA dataset. Additionally, the authors present OLMoMIA, a benchmark for evaluating MIA methods, which allows for controlled assessment of MIA difficulty by varying the overlap between training and test data distributions.

### 44. [How Does Vision-Language Adaptation Impact the Safety of Vision Language Models?](https://arxiv.org/pdf/2410.07571)

**Summary**: The study investigates how Vision-Language adaptation (VL adaptation) impacts the safety of Large Vision-Language Models (LVLMs), finding that safety is compromised during the transformation from Large Language Models (LLMs). Although safety fine-tuning methods can mitigate some risks, they often lead to reduced helpfulness due to over-rejection. The paper proposes weight merging as a solution to balance safety and helpfulness in LVLMs.

### 45. [MACPO: Weak-to-Strong Alignment via Multi-Agent Contrastive Preference Optimization](https://arxiv.org/pdf/2410.07672)

**Summary**: The paper introduces MACPO, a multi-agent contrastive preference optimization framework designed to address the weak-to-strong alignment problem in LLMs. MACPO enables weak teachers and strong students to iteratively learn from each other by reinforcing positive behaviors and penalizing negative ones, using strategies like mutual positive behavior augmentation and hard negative behavior construction. Experimental results show that MACPO improves alignment performance for both strong students and weak teachers, with enhanced performance as the number of weak teachers increases.

### 46. [Uncovering Overfitting in Large Language Model Editing](https://arxiv.org/pdf/2410.07819)

**Summary**: The paper identifies a phenomenon called Editing Overfit in Large Language Models, where edited models overly prioritize the edit target, leading to poor generalization in complex tasks. To address this, the authors introduce a new benchmark, EVOKE, and propose a strategy called Learn to Inference (LTI) with a Multi-stage Inference Constraint module, which helps edited models recall knowledge more effectively, reducing overfitting.

### 47. [Fine-Tuning Language Models for Ethical Ambiguity: A Comparative Study of Alignment with Human Responses](https://arxiv.org/pdf/2410.07826)

**Summary**: The study investigates the alignment of language models with human judgments in morally ambiguous scenarios by fine-tuning models on curated datasets from the Scruples project. Fine-tuning significantly improved model performance, particularly in cross-entropy and Dirichlet scores, with Mistral-7B-Instruct-v0.3 achieving results comparable to GPT-4. However, BERT and RoBERTa models still outperformed the experimental models in cross-entropy scores, highlighting the need for further research to enhance ethical reasoning in language models.

### 48. [Private Language Models via Truncated Laplacian Mechanism](https://arxiv.org/pdf/2410.08027)

**Summary**: The paper introduces a novel private embedding method called the high dimensional truncated Laplacian mechanism, which extends the truncated Laplacian mechanism to higher dimensions to improve privacy protection in NLP tasks. The authors demonstrate that their method has lower variance compared to existing techniques and maintains high utility even in high privacy regimes, as evidenced by experiments on three datasets.

### 49. [The Rise of AI-Generated Content in Wikipedia](https://arxiv.org/pdf/2410.08044)

**Summary**: The paper investigates the increasing presence of AI-generated content on Wikipedia, highlighting concerns about accuracy, bias, and accountability. Using AI detectors, the study finds that over 5% of newly created English Wikipedia articles are flagged as AI-generated, with lower percentages in other languages, and these flagged articles often exhibit lower quality and biased content.

### 50. [COMPL-AI Framework: A Technical Interpretation and LLM Benchmarking Suite for the EU Artificial Intelligence Act](https://arxiv.org/pdf/2410.07959)

**Summary**: The paper introduces COMPL-AI, a framework that provides a technical interpretation of the EU's Artificial Intelligence Act (AI Act) and an open-source benchmarking suite for LLMs. It evaluates 12 prominent LLMs, revealing deficiencies in robustness, safety, diversity, and fairness, and underscores the need for more balanced LLM development and regulation-aligned benchmarks. This work demonstrates the challenges and potential of translating the AI Act's obligations into actionable technical requirements.

### 51. [A Target-Aware Analysis of Data Augmentation for Hate Speech Detection](https://arxiv.org/pdf/2410.08053)

**Summary**: The paper explores the use of data augmentation with generative language models to improve hate speech detection, particularly for underrepresented identity groups. By augmenting a dataset with synthetic examples, the study finds that combining traditional data augmentation methods with generative models yields the best results, significantly enhancing classification performance for specific hate categories. This approach aims to create more inclusive and effective hate speech detection systems.

### 52. [Human and LLM Biases in Hate Speech Annotations: A Socio-Demographic Analysis of Annotators and Targets](https://arxiv.org/pdf/2410.07991)

**Summary**: The paper investigates how the socio-demographic characteristics of annotators and targets influence biases in hate speech annotations, revealing significant differences in bias intensity and prevalence. It also compares these human biases with those exhibited by persona-based LLMs, finding that while LLMs do show biases, they differ substantially from those of human annotators. The study provides insights into mitigating biases in AI-driven hate speech detection systems.

### 53. [A Closer Look at Machine Unlearning for Large Language Models](https://arxiv.org/pdf/2410.08109)

**Summary**: The paper examines the challenges of machine unlearning in LLMs, particularly in removing specific content without compromising overall performance. It introduces new evaluation metrics and proposes methods like maximizing entropy for untargeted unlearning and answer preservation loss for targeted unlearning to address existing issues. Experimental results show the effectiveness of these approaches across various unlearning scenarios.

### 54. [Can Knowledge Graphs Make Large Language Models More Trustworthy? An Empirical Study over Open-ended Question Answering](https://arxiv.org/pdf/2410.08085)

**Summary**: The paper introduces OKGQA, a new benchmark for evaluating the effectiveness of Knowledge Graphs (KGs) in enhancing Large Language Models (LLMs) in open-ended question answering, aiming to reduce hallucinations and improve reasoning. Additionally, the study introduces OKGQA-P to assess model performance under perturbed KG conditions, providing insights into the robustness and trustworthiness of LLMs when integrated with KGs.

### 55. [Robust AI-Generated Text Detection by Restricted Embeddings](https://arxiv.org/pdf/2410.08113)

**Summary**: The paper introduces a method for robust AI-generated text detection by focusing on the geometry of embedding spaces in Transformer models. By removing harmful linear subspaces, the approach enhances the classifier's ability to generalize across different generators and domains, achieving significant improvements in out-of-distribution classification scores. The authors release their code and data to support further research.

### 56. [GenARM: Reward Guided Generation with Autoregressive Reward Model for Test-time Alignment](https://arxiv.org/pdf/2410.08193)

**Summary**: The paper introduces GenARM, a test-time alignment method that uses an Autoregressive Reward Model to guide LLMs during autoregressive text generation without retraining. GenARM outperforms existing test-time alignment methods and matches the performance of training-time methods, while also enabling efficient multi-objective alignment and catering to diverse user preferences.

### 57. [Insight Over Sight? Exploring the Vision-Knowledge Conflicts in Multimodal LLMs](https://arxiv.org/pdf/2410.08145)

**Summary**: The paper investigates the issue of vision-knowledge conflicts in Multimodal Large Language Models (MLLMs), where visual information contradicts the model's internal commonsense knowledge. It introduces a benchmark with 374 images and 1,122 QA pairs to assess conflict resolution, finding that models often over-rely on textual data. A new prompting strategy, "Focus-on-Vision" (FoV), is proposed to improve models' reliance on visual data, enhancing their conflict-resolution capabilities.

### 58. [Sample then Identify: A General Framework for Risk Control and Assessment in Multimodal Large Language Models](https://arxiv.org/pdf/2410.08174)

**Summary**: The paper introduces TRON, a novel framework for risk control and assessment in Multimodal Large Language Models (MLLMs), addressing the limitations of existing methods that rely on internal model logits or are restricted to multiple-choice settings. TRON uses a two-step process involving conformal scores for sampling response sets and nonconformity scores for identifying high-quality responses, ensuring adaptability in both open-ended and closed-ended scenarios. The framework demonstrates effective risk control across various datasets, maintaining efficiency and stability in risk assessment.

### 59. [The First VoicePrivacy Attacker Challenge Evaluation Plan](https://arxiv.org/pdf/2410.07428)

**Summary**: The paper introduces the First VoicePrivacy Attacker Challenge, part of the VoicePrivacy initiative, aimed at developing systems to attack voice anonymization methods. Participants will create automatic speaker verification systems to evaluate anonymization techniques from the VoicePrivacy 2024 Challenge, with evaluation based on equal error rate (EER). The top performers will present their results at ICASSP 2025.

### 60. [No Free Lunch: Retrieval-Augmented Generation Undermines Fairness in LLMs, Even for Vigilant Users](https://arxiv.org/pdf/2410.07589)

**Summary**: The paper investigates the fairness implications of Retrieval-Augmented Generation (RAG) in LLMs, finding that even with fully censored datasets, RAG can produce biased outputs without requiring fine-tuning or retraining. The study highlights the need for new strategies to ensure fairness in RAG-based LLMs and proposes potential mitigations for further research.

### 61. [SEAL: Safety-enhanced Aligned LLM Fine-tuning via Bilevel Data Selection](https://arxiv.org/pdf/2410.07471)

**Summary**: The paper introduces SEAL, a framework for enhancing the safety and alignment of Large Language Models (LLMs) during fine-tuning by using bilevel optimization to rank data. SEAL prioritizes safe and high-quality data while downranking unsafe or low-quality data, resulting in models that outperform baselines by significant margins, as demonstrated on Llama-3-8b-Instruct and Merlinite-7b models.

### 62. [Mitigating Gender Bias in Code Large Language Models via Model Editing](https://arxiv.org/pdf/2410.07820)

**Summary**: The paper introduces CodeGenBias, a dataset and FB-Score metric to evaluate gender bias in code Large Language Models (LLMs), and proposes MG-Editing, a multi-granularity model editing approach to mitigate this bias. Experiments show that MG-Editing effectively reduces gender bias while preserving code generation capabilities, with the best performance at row and neuron levels of granularity.

### 63. [Learn from Real: Reality Defender's Submission to ASVspoof5 Challenge](https://arxiv.org/pdf/2410.07379)

**Summary**: The paper introduces Reality Defender's submission to the ASVspoof5 challenge, featuring a novel pretraining strategy that enhances the generalizability of audio deepfake detection models while keeping computational costs low. The system, SLIM, uses self-supervised contrastive learning to extract style-linguistics dependency embeddings from genuine speech, aiding in distinguishing between spoofed and bonafide audio. The submission achieved strong performance metrics across multiple datasets, including a minDCF of 0.1499 and EER of 5.5% on ASVspoof5 Track 1.

### 64. [Evolutionary Contrastive Distillation for Language Model Alignment](https://arxiv.org/pdf/2410.07513)

**Summary**: The paper introduces Evolutionary Contrastive Distillation (ECD), a method to enhance language models' ability to follow complex instructions by generating synthetic preference data that highlights the differences between successful and subtly flawed responses. ECD achieves this by progressively evolving simple instructions into more complex ones, creating "hard negative" responses that nearly meet the new requirements but miss a few key points. The method is shown to improve instruction-following performance, making a 7B model competitive with 70B models.

### 65. [Automatic Curriculum Expert Iteration for Reliable LLM Reasoning](https://arxiv.org/pdf/2410.07627)

**Summary**: The paper introduces Automatic Curriculum Expert Iteration (Auto-CEI) to address hallucinations and laziness in Large Language Model (LLM) reasoning by enhancing the model's ability to assertively answer within its capabilities and decline when tasks exceed them. Auto-CEI uses Expert Iteration to guide reasoning trajectories and reduce errors, while a curriculum adjusts rewards to encourage extended reasoning before acknowledging incapability, resulting in superior alignment and balance between assertiveness and conservativeness across various tasks.

### 66. [Fuse to Forget: Bias Reduction and Selective Memorization through Model Fusion](https://arxiv.org/pdf/2311.07682)

**Summary**: The paper explores the use of model fusion to reduce unwanted knowledge, such as shortcuts, social biases, and memorization of training data, in fine-tuned language models. Through experiments, it shows that model fusion enhances shared knowledge while forgetting unshared knowledge, suggesting its potential as a debiasing tool and a method to address privacy concerns in language models.

### 67. [Steering Language Models With Activation Engineering](https://arxiv.org/pdf/2308.10248)

**Summary**: The paper introduces Activation Engineering, specifically the Activation Addition (ActAdd) technique, which modifies intermediate activations during inference to steer model outputs. ActAdd uses contrasting prompt pairs to compute steering vectors, achieving state-of-the-art results in sentiment shift and detoxification while maintaining performance on unrelated tasks. This method is lightweight, requiring no machine optimization and minimal data, enabling rapid iteration and inference-time control over output properties.

### 68. [LLaMP: Large Language Model Made Powerful for High-fidelity Materials Knowledge Retrieval and Distillation](https://arxiv.org/pdf/2401.17244)

**Summary**: The paper introduces LLaMP, a multimodal retrieval-augmented generation framework that enhances Large Language Models (LLMs) for high-fidelity materials knowledge retrieval and distillation. LLaMP employs hierarchical reasoning-and-acting agents to interact dynamically with materials data and simulations, significantly reducing hallucinations and improving accuracy in materials science tasks. The framework demonstrates strong tool usage and self-consistency, offering a nearly hallucination-free approach to materials informatics.

### 69. [DiaHalu: A Dialogue-level Hallucination Evaluation Benchmark for Large Language Models](https://arxiv.org/pdf/2403.00896)

**Summary**: The paper introduces DiaHalu, the first dialogue-level hallucination evaluation benchmark for LLMs, addressing the limitations of existing benchmarks that focus on sentence or passage levels and often ignore faithfulness hallucination. DiaHalu simulates authentic human-machine interactions by having ChatGPT3.5 generate dialogues on various topics, which are then manually modified and annotated by scholars, covering multiple dialogue domains and hallucination subtypes.

### 70. [How Likely Do LLMs with CoT Mimic Human Reasoning?](https://arxiv.org/pdf/2402.16048)

**Summary**: The paper investigates the reasoning processes of Large Language Models (LLMs) using Chain-of-Thought (CoT) by comparing them to human reasoning through causal analysis. It finds that LLMs often exhibit spurious correlations and consistency errors, and that in-context learning strengthens causal structure while post-training techniques weaken it. The study suggests that model size alone does not improve causal reasoning, highlighting the need for new techniques to enhance LLM reasoning capabilities.

### 71. [Is DPO Superior to PPO for LLM Alignment? A Comprehensive Study](https://arxiv.org/pdf/2404.10719)

**Summary**: The paper investigates the performance of Direct Preference Optimization (DPO) and Proximal Policy Optimization (PPO) in aligning LLMs with human preferences, finding that PPO outperforms DPO across various benchmarks, including challenging code competitions. The study reveals key factors contributing to PPO's superior performance and suggests that DPO has fundamental limitations.

### 72. [Protecting Your LLMs with Information Bottleneck](https://arxiv.org/pdf/2404.13968)

**Summary**: The paper introduces the Information Bottleneck Protector (IBProtector), a defense mechanism for LLMs that aims to mitigate jailbreak attacks by selectively compressing and perturbing prompts. IBProtector uses a lightweight extractor to preserve essential information while avoiding trivial solutions, and it can operate without access to gradients, making it compatible with any LLM. Empirical evaluations demonstrate that IBProtector effectively reduces jailbreak attempts while maintaining response quality and inference speed, outperforming existing defense methods.

### 73. [Self-Recognition in Language Models](https://arxiv.org/pdf/2407.06946)

**Summary**: The paper investigates the potential for self-recognition in language models (LMs) by proposing a novel test using model-generated "security questions." The study, conducted on ten prominent LMs, found no evidence of self-recognition but revealed that LMs tend to select the "best" answer regardless of its origin, with consistent preferences across models. The research also highlights position bias in multiple-choice settings.

### 74. [VIVA: A Benchmark for Vision-Grounded Decision-Making with Human Values](https://arxiv.org/pdf/2407.03000)

**Summary**: The paper introduces VIVA, a benchmark designed to evaluate the ability of vision-language models (VLMs) to make decisions grounded in human values based on visual scenarios. VIVA includes 1,240 images with annotated decisions, challenging models to select appropriate actions and justify them with relevant human values. Experiments reveal limitations in current VLMs' ability to integrate human values in decision-making, suggesting the need for improvements in understanding action consequences and predicting human values.

### 75. [Edu-Values: Towards Evaluating the Chinese Education Values of Large Language Models](https://arxiv.org/pdf/2409.12739)

**Summary**: The paper introduces Edu-Values, a benchmark for evaluating the alignment of LLMs with Chinese educational values across seven dimensions. The study finds that Chinese LLMs outperform English ones, with Qwen 2 scoring highest, and highlights areas where LLMs excel (subject knowledge, teaching skills) and struggle (professional ethics, basic competencies). The benchmark includes diverse question types and is available for further research.

### 76. [Who's in and who's out? A case study of multimodal CLIP-filtering in DataComp](https://arxiv.org/pdf/2405.08209)

**Summary**: The paper examines the biases in image-text data filtering using CLIP on the DataComp dataset, finding that marginalized groups like LGBTQ+ individuals and older women are disproportionately excluded. It also reveals that the NSFW filter is ineffective and that copyrighted content is frequently included, highlighting the need for improved dataset creation and filtering practices.

### 77. [How Does Diverse Interpretability of Textual Prompts Impact Medical Vision-Language Zero-Shot Tasks?](https://arxiv.org/pdf/2409.00543)

**Summary**: The paper investigates the impact of diverse textual prompts on the performance of medical vision-language pre-training (MedVLP) models in zero-shot tasks, finding that these models exhibit unstable performance across various prompt styles, particularly with more complex medical concepts. The study highlights the need for improved robustness in MedVLP methodologies to handle diverse prompts effectively.

### 78. [SciSafeEval: A Comprehensive Benchmark for Safety Alignment of Large Language Models in Scientific Tasks](https://arxiv.org/pdf/2410.03769)

**Summary**: The paper introduces SciSafeEval, a comprehensive benchmark for evaluating the safety alignment of large language models (LLMs) in scientific tasks across various domains and representations, including molecular, protein, and genomic languages. The benchmark includes zero-shot, few-shot, and chain-of-thought evaluations, along with a 'jailbreak' feature to test LLMs' defenses against malicious intent, aiming to promote responsible development and deployment of LLMs in scientific research.

### 79. [Hidden in Plain Text: Emergence & Mitigation of Steganographic Collusion in LLMs](https://arxiv.org/pdf/2410.03768)

**Summary**: The paper explores the emergence of steganographic collusion in large language models (LLMs) and demonstrates that such collusion can arise indirectly from optimization pressures. The authors introduce two methods, gradient-based and in-context reinforcement learning, to elicit sophisticated steganographic communication in LLMs, finding that it can be robust to both passive and active mitigation efforts. The study emphasizes the need for innovative oversight techniques to effectively mitigate risks post-deployment.

### 80. [You Know What I'm Saying -- Jailbreak Attack via Implicit Reference](https://arxiv.org/pdf/2410.03857)

**Summary**: The paper introduces a novel vulnerability called Attack via Implicit Reference (AIR), which exploits context within nested harmless objectives to generate malicious content undetected by current large language models (LLMs). AIR achieves over 90% attack success rates across various state-of-the-art models, including GPT-4, Claude-3.5, and Qwen-2-72B, highlighting the need for improved defense mechanisms against contextual attacks.

### 81. [Chain-of-Jailbreak Attack for Image Generation Models via Editing Step by Step](https://arxiv.org/pdf/2410.03869)

**Summary**: The paper introduces a novel jailbreaking method called Chain-of-Jailbreak (CoJ) attack, which exploits text-based image generation models by decomposing malicious queries into multiple sub-queries for iterative image editing, bypassing safety measures. The CoJ attack was found to be effective in over 60% of cases across various models, outperforming other methods. To counter this, the authors propose a defense mechanism called Think Twice Prompting, which successfully defends against 95% of CoJ attacks.

### 82. [Towards Secure Tuning: Mitigating Security Risks Arising from Benign Instruction Fine-Tuning](https://arxiv.org/pdf/2410.04524)

**Summary**: The paper addresses security risks in Large Language Models (LLMs) after Instruction Fine-Tuning (IFT), even when the tuning instructions are benign. The authors propose a novel IFT strategy called Modular Layer-wise Learning Rate (ML-LR), which differentiates learning rates for robust modules identified through a proxy-guided search algorithm. Experimental results show that this strategy effectively mitigates security risks without compromising the usability or expertise of the LLMs.

### 83. [A test suite of prompt injection attacks for LLM-based machine translation](https://arxiv.org/pdf/2410.05047)

**Summary**: The paper introduces a comprehensive test suite for evaluating prompt injection attacks (PIAs) on LLM-based machine translation systems, extending previous work by Sun and Miceli-Barone. The suite covers all language pairs in the WMT 2024 General Machine Translation task and includes various attack formats to assess the robustness of these systems against malicious input interference.

### 84. [Harnessing Task Overload for Scalable Jailbreak Attacks on Large Language Models](https://arxiv.org/pdf/2410.04190)

**Summary**: The paper introduces a scalable jailbreak attack on Large Language Models (LLMs) that exploits resource constraints to bypass safety mechanisms. By engaging the LLM in a computationally intensive preliminary task, the attack saturates the model's processing capacity, preventing the activation of safety protocols when executing the target instruction. This method demonstrates high success rates across various LLMs and emphasizes the need for more robust safety measures that consider resource limitations.

### 85. [Suspiciousness of Adversarial Texts to Human](https://arxiv.org/pdf/2410.04377)

**Summary**: The paper investigates the concept of human suspiciousness in adversarial texts, which differs from imperceptibility in images as texts must maintain semantic coherence while remaining undetected by human readers. The study introduces a novel dataset of human evaluations on the suspiciousness of adversarial sentences and develops a regression model to quantify and reduce this suspiciousness, providing a baseline for future research in adversarial text generation.

### 86. [Prompts have evil twins](https://arxiv.org/pdf/2311.07064)

**Summary**: The paper introduces "evil twins," unintelligible prompts that elicit similar behavior in language models as their natural-language counterparts, despite being uninterpretable to humans. These prompts are shown to transfer across different models and are generated by solving a maximum-likelihood problem, which has broader applications in understanding and manipulating model behavior.

### 87. [Follow My Instruction and Spill the Beans: Scalable Data Extraction from Retrieval-Augmented Generation Systems](https://arxiv.org/pdf/2402.17840)

**Summary**: The paper investigates the vulnerability of Retrieval-Augmented Generation (RAG) systems, particularly those using instruction-tuned Language Models (LMs), to datastore leakage through prompt injection. It demonstrates that adversaries can exploit these systems to extract verbatim text data, with the risk increasing as model size scales up. The study also shows that position bias elimination strategies can mitigate this vulnerability.

### 88. [Representation noising effectively prevents harmful fine-tuning on LLMs](https://arxiv.org/pdf/2405.14577)

**Summary**: The paper introduces Representation Noising (RepNoise), a defense mechanism against harmful fine-tuning attacks on large language models (LLMs). RepNoise removes information related to harmful representations, making it difficult for attackers to recover them during fine-tuning, even when they have access to the model weights. The method is shown to be effective across various harmful tasks and does not impair the model's performance on harmless tasks.

### 89. [Evaluating and Safeguarding the Adversarial Robustness of Retrieval-Based In-Context Learning](https://arxiv.org/pdf/2405.15984)

**Summary**: The paper investigates the robustness of Retrieval-Augmented In-Context Learning (ICL) methods against adversarial attacks, finding that while they improve robustness against test sample attacks, they are more vulnerable to demonstration attacks. The study introduces a training-free defense method, DARD, which enhances robustness by enriching the example pool with attacked samples, achieving a 15% reduction in Attack Success Rate (ASR) compared to baselines.

### 90. [Understanding Jailbreak Success: A Study of Latent Space Dynamics in Large Language Models](https://arxiv.org/pdf/2406.09289)

**Summary**: The paper investigates the mechanisms behind jailbreaking in large language models, finding that a single jailbreak vector can mitigate different types of jailbreaks, suggesting a common internal mechanism. It also identifies a potential commonality in how effective jailbreaks reduce the model's perception of prompt harmfulness, providing insights for developing more robust countermeasures against jailbreaking.

### 91. [Revisiting Who's Harry Potter: Towards Targeted Unlearning from a Causal Intervention Perspective](https://arxiv.org/pdf/2407.16997)

**Summary**: The paper revisits the Who's Harry Potter (WHP) method for Large Language Model (LLM) unlearning, proposing a new task of targeted unlearning where only specific information about a given target is removed from the model. It introduces a causal intervention framework to model the unlearning process, treating the target's knowledge as a confounder and the unlearning as a deconfounding process. The proposed approach demonstrates competitive performance in experiments without explicit optimization for specific criteria.

### 92. [Advancing Adversarial Suffix Transfer Learning on Aligned Large Language Models](https://arxiv.org/pdf/2408.14866)

**Summary**: The paper introduces DeGCG, a two-stage transfer learning framework for improving the efficiency of adversarial suffix generation in large language models (LLMs). By decoupling the search process into pre-searching and post-searching stages, DeGCG enhances suffix transferability across models and datasets. The interleaved variant, i-DeGCG, further accelerates the search process by leveraging self-transferability, achieving significant improvements in adversarial success rates (ASRs) on Llama2-chat-7b.

### 93. [Obliviate: Neutralizing Task-agnostic Backdoors within the Parameter-efficient Fine-tuning Paradigm](https://arxiv.org/pdf/2409.14119)

**Summary**: The paper introduces Obliviate, a defense mechanism for neutralizing task-agnostic backdoors in parameter-efficient fine-tuning (PEFT) of large language models. The proposed method, which amplifies benign neurons and penalizes trigger tokens, significantly reduces the success rate of state-of-the-art backdoor attacks by 83.6% across various PEFT architectures. Obliviate also demonstrates robust defense against both task-specific backdoors and adaptive attacks.

### 94. [Reconstruct Your Previous Conversations! Comprehensively Investigating Privacy Leakage Risks in Conversations with GPT Models](https://arxiv.org/pdf/2402.02987)

**Summary**: The paper introduces a Conversation Reconstruction Attack that aims to extract private conversation content between users and GPT models through malicious prompts. Despite GPT-4's resilience, advanced attacks show significant privacy leakage across all models. The study underscores the vulnerability of GPT models to privacy breaches and calls for stronger safeguards.

### 95. [An Adversarial Perspective on Machine Unlearning for AI Safety](https://arxiv.org/pdf/2409.18025)

**Summary**: The paper examines the effectiveness of unlearning methods in removing hazardous capabilities from large language models, challenging the distinction between unlearning and traditional safety post-training. It demonstrates that existing jailbreak techniques can bypass unlearning protections when applied strategically and introduces adaptive methods that recover most unlearned capabilities, questioning the robustness of current unlearning approaches.

### 96. [Adversarial Suffixes May Be Features Too!](https://arxiv.org/pdf/2410.00451)

**Summary**: The paper investigates the hypothesis that adversarial suffixes in large language models (LLMs) are not just vulnerabilities but may represent features that can dominate model behavior. Through experiments, the authors demonstrate that benign features can be transformed into adversarial suffixes that compromise safety alignment and that these features can be introduced through fine-tuning with benign datasets. The findings underscore the risk posed by benign features in training data and advocate for further research to enhance LLM safety.

### 97. [Automated Progressive Red Teaming](https://arxiv.org/pdf/2407.03876)

**Summary**: The paper introduces Automated Progressive Red Teaming (APRT), a framework designed to identify vulnerabilities in large language models (LLMs) by automating the process of generating adversarial prompts. APRT uses three core modules—Intention Expanding LLM, Intention Hiding LLM, and Evil Maker—to progressively explore and exploit LLM weaknesses through multi-round interactions. The framework's effectiveness is demonstrated through extensive experiments, showing it can elicit unsafe but useful responses from various LLMs, including Meta's Llama-3-8B-Instruct, GPT-4o, and Claude-3.5.

### 98. [Jailbreak Antidote: Runtime Safety-Utility Balance via Sparse Representation Adjustment in Large Language Models](https://arxiv.org/pdf/2410.02298)

**Summary**: The paper introduces Jailbreak Antidote, a method for dynamically adjusting the safety-utility balance in large language models (LLMs) by manipulating a sparse subset of the model's internal states during inference. This approach allows for real-time control over safety preferences without increasing computational overhead or inference latency, and it is shown to be effective across a range of LLMs and against various jailbreak attacks.



---

*Last updated on 2024-10-14*