# Awesome-LLM-Jailbreak

## Awesome LLM Jailbreak

Welcome to the **Awesome LLM Jailbreak** repository! This project curates a list of high-quality resources related to LLM Jailbreak, including research papers, tools, libraries, and more.

## Daily Updates

### 2024-10-15

### 1. [Recent advancements in LLM Red-Teaming: Techniques, Defenses, and Ethical Considerations](https://arxiv.org/pdf/2410.09097)

**Summary**: The paper reviews recent advancements in red-teaming Large Language Models (LLMs), focusing on attack strategies such as gradient-based optimization, reinforcement learning, and prompt engineering, and their impact on LLM security. It emphasizes the importance of developing robust defense mechanisms to counter these vulnerabilities, aiming to enhance the reliability and safety of LLMs.

### 2. [Knowledge-Augmented Reasoning for EUAIA Compliance and Adversarial Robustness of LLMs](https://arxiv.org/pdf/2410.09078)

**Summary**: The paper introduces a functional architecture to address the dual challenge of ensuring compliance with the EU AI Act (EUAIA) and achieving adversarial robustness in LLMs. By integrating a reasoning layer based on knowledge augmentation, the proposed system aims to support developers and auditors in verifying both compliance and robustness, thereby enhancing the trustworthiness of deployed LLMs in the EU.

### 3. [M3Hop-CoT: Misogynous Meme Identification with Multimodal Multi-hop Chain-of-Thought](https://arxiv.org/pdf/2410.09220)

**Summary**: The paper introduces M3Hop-CoT, a novel framework for identifying misogynous memes by integrating multimodal data and multi-hop Chain-of-Thought reasoning. The framework combines a CLIP-based classifier with a multimodal CoT module to enhance detection by considering cultural diversity, emotions, and contextual knowledge. Evaluations on the SemEval-2022 Task 5 dataset and other benchmarks demonstrate M3Hop-CoT's superior performance in macro-F1 score and generalizability across different datasets.

### 4. [Impeding LLM-assisted Cheating in Introductory Programming Assignments via Adversarial Perturbations](https://arxiv.org/pdf/2410.09318)

**Summary**: The paper explores the use of adversarial perturbations to hinder the performance of LLMs in generating code for introductory programming assignments, aiming to prevent cheating. A user study found that combined perturbations reduced the average correctness score by 77%, with the effectiveness of these perturbations influenced by their detectability.

### 5. [Sui Generis: Large Language Models for Authorship Attribution and Verification in Latin](https://arxiv.org/pdf/2410.09245)

**Summary**: The paper examines the effectiveness of Large Language Models (LLMs) in authorship attribution and verification for Latin texts from the Patristic Era, finding that LLMs can perform well in zero-shot scenarios but are susceptible to semantic confusion. The study highlights the challenges in guiding LLMs to make nuanced and explainable decisions, particularly when compared to their performance in high-resource modern languages.

### 6. [Nudging: Inference-time Alignment via Model Collaboration](https://arxiv.org/pdf/2410.09300)

**Summary**: The paper introduces "nudging," a training-free algorithm that aligns large language models at inference time using a smaller aligned model. By leveraging the base model's uncertainty in generating certain stylistic tokens, nudging effectively steers the output towards desired directions, achieving performance comparable to or better than fully aligned models without additional training. This method enables modular collaboration between different model families, offering a computationally efficient solution to model alignment.

### 7. [Keys to Robust Edits: from Theoretical Insights to Practical Advances](https://arxiv.org/pdf/2410.09338)

**Summary**: The paper investigates the limitations of current knowledge editing techniques in LLMs, particularly their lack of robustness in handling long contexts and paraphrased subjects. Through theoretical analysis, the authors identify key-value modeling issues and propose a novel 'group discussion' model to improve robustness. They introduce the Robust Edit Pathway (REP) to separate editing keys from LLM representations, demonstrating significant improvements in robustness across multiple metrics with minimal performance trade-offs.

### 8. [FB-Bench: A Fine-Grained Multi-Task Benchmark for Evaluating LLMs' Responsiveness to Human Feedback](https://arxiv.org/pdf/2410.09412)

**Summary**: The paper introduces FB-Bench, a comprehensive benchmark designed to evaluate the responsiveness of Large Language Models (LLMs) to human feedback in real-world scenarios. FB-Bench includes 734 curated samples across eight task types, five response deficiencies, and nine feedback types, revealing significant variations in LLM performance. The study highlights the importance of considering task complexity, feedback type, and response deficiencies in assessing LLM capabilities.

### 9. [Solving the Challenge Set without Solving the Task: On Winograd Schemas as a Test of Pronominal Coreference Resolution](https://arxiv.org/pdf/2410.09448)

**Summary**: The paper challenges the assumption that high performance on the Winograd Schema Challenge (WSC) necessarily indicates strong performance in resolving pronominal coreference in general. It demonstrates that while language models excel on WSC, they struggle with simpler coreference tasks in other datasets like OntoNotes. The authors propose an ensemble method combining language models with task-specific systems to improve overall coreference resolution accuracy and highlight the need for comprehensive evaluation across diverse datasets.

### 10. [CollabEdit: Towards Non-destructive Collaborative Knowledge Editing](https://arxiv.org/pdf/2410.09508)

**Summary**: The paper introduces CollabEdit, a non-destructive collaborative knowledge editing framework for LLMs, addressing challenges like knowledge overlap, conflict, and forgetting. By using a novel model merging mechanism, CollabEdit mimics global knowledge editing while preventing performance drops, outperforming destructive baselines in experiments on two datasets.

### 11. [Are You Human? An Adversarial Benchmark to Expose LLMs](https://arxiv.org/pdf/2410.09569)

**Summary**: The paper introduces an adversarial benchmark to detect Large Language Models (LLMs) in conversations by using implicit and explicit challenges. The benchmark, tested on nine leading models, shows that explicit challenges are more effective in identifying LLMs, while implicit challenges are less so. The study also highlights the prevalence of human participants using LLMs, emphasizing the need for reliable detection methods in critical interactions.

### 12. [Extended Japanese Commonsense Morality Dataset with Masked Token and Label Enhancement](https://arxiv.org/pdf/2410.09564)

**Summary**: The paper introduces the Extended Japanese Commonsense Morality (eJCM) dataset, which expands the original JCM dataset from 13,975 to 31,184 sentences using a novel method called Masked Token and Label Enhancement (MTLE). This method improves the dataset's cultural relevance and complexity, leading to a model trained on eJCM achieving higher F1 scores in moral reasoning tasks, particularly in culturally nuanced scenarios, compared to other models and augmentation techniques.

### 13. [Quebec Automobile Insurance Question-Answering With Retrieval-Augmented Generation](https://arxiv.org/pdf/2410.09623)

**Summary**: The paper investigates the use of Retrieval-Augmented Generation (RAG) with a state-of-the-art Large Language Model (LLM) to answer Quebec automobile insurance questions, leveraging a specialized corpus. It finds that while the RAG approach generally improves response quality, there is a significant risk of misinformation, with 5% to 13% of answers containing false statements that could mislead customers.

### 14. [Honest AI: Fine-Tuning "Small" Language Models to Say "I Don't Know", and Reducing Hallucination in RAG](https://arxiv.org/pdf/2410.09699)

**Summary**: The paper introduces Honest AI, a strategy to fine-tune smaller language models to reduce hallucination by teaching them to say "I don't know." It explores various approaches, including Retrieval-Augmented Generation (RAG) and fine-tuning, finding that a hybrid approach combining both methods performs best. The study highlights the effectiveness of smaller models, emphasizing resource efficiency.

### 15. [Taming Overconfidence in LLMs: Reward Calibration in RLHF](https://arxiv.org/pdf/2410.09724)

**Summary**: The paper addresses the issue of overconfidence in Large Language Models (LLMs) trained with Reinforcement Learning from Human Feedback (RLHF), revealing that RLHF tends to induce verbalized overconfidence in model responses. To mitigate this, the authors propose two PPO variants, PPO-M and PPO-C, which calibrate reward models and reward calculations, respectively, to better align confidence with response quality. Experimental results show that these methods reduce calibration error without compromising model performance.

### 16. ['Quis custodiet ipsos custodes?' Who will watch the watchmen? On Detecting AI-generated peer-reviews](https://arxiv.org/pdf/2410.09770)

**Summary**: The paper introduces two models, Term Frequency (TF) and Review Regeneration (RR), to detect AI-generated peer-reviews, addressing concerns about the integrity of the peer-review process in academic publishing. The TF model identifies repeated tokens by AI, while the RR model detects similar outputs from ChatGPT upon re-prompting. Both models are tested against token attacks and paraphrasing, with the RR model proving more robust. The study concludes that these methods outperform other AI text detectors and makes its resources publicly available.

### 17. [MisinfoEval: Generative AI in the Era of "Alternative Facts"](https://arxiv.org/pdf/2410.09949)

**Summary**: The paper introduces MisinfoEval, a framework for evaluating large language model (LLM) based interventions to combat misinformation on social media. Experiments in simulated environments show that LLM-based interventions significantly improve users' accuracy in identifying misinformation, with personalized explanations tailored to users' demographics and beliefs being particularly effective, increasing accuracy by up to 41.72%.

### 18. [When Neutral Summaries are not that Neutral: Quantifying Political Neutrality in LLM-Generated News Summaries](https://arxiv.org/pdf/2410.09978)

**Summary**: The study investigates the political neutrality of LLMs in generating summaries of polarizing news articles, focusing on five key US political issues. It finds a consistent pro-Democratic bias in several LLMs, particularly in gun control and healthcare, with significant vocabulary overlap in the generated summaries that align with Democratic perspectives. The findings are particularly relevant given the upcoming US elections.

### 19. [RMB: Comprehensively Benchmarking Reward Models in LLM Alignment](https://arxiv.org/pdf/2410.09893)

**Summary**: The paper introduces RMB, a comprehensive benchmark for evaluating reward models (RMs) in large language model (LLM) alignment, addressing limitations in current evaluation methods. RMB covers 49 real-world scenarios and employs both pairwise and Best-of-N evaluations to better reflect RM effectiveness. The study reveals generalization defects in state-of-the-art RMs and explores the potential of generative RMs, while also examining the impact of evaluation criteria and instructing methods.

### 20. [Evaluating Gender Bias of LLMs in Making Morality Judgements](https://arxiv.org/pdf/2410.09992)

**Summary**: The paper evaluates gender bias in LLMs when making moral judgments, using a new dataset called GenMO. It finds that despite safety measures, all tested models, including GPT-3.5-turbo and Llama 3, exhibit significant gender bias, often favoring female characters. The study also examines how model parameters influence bias and explores real-world scenarios where LLMs reveal biased moral decisions.

### 21. [Safety-Aware Fine-Tuning of Large Language Models](https://arxiv.org/pdf/2410.10014)

**Summary**: The paper introduces a Safety-Aware Fine-Tuning (SAFT) framework to automatically detect and remove harmful data during the fine-tuning of Large Language Models (LLMs), addressing the challenge of labor-intensive manual filtering. The framework leverages a scoring function based on subspace information to reduce harmfulness by up to 27.8%, demonstrating its effectiveness across various models and contamination rates, and validating its practical applicability in real-world scenarios.

### 22. [Diagnosing Hate Speech Classification: Where Do Humans and Machines Disagree, and Why?](https://arxiv.org/pdf/2410.10153)

**Summary**: The study investigates discrepancies between human and machine hate speech classification using cosine similarity, embedding regression, and manual re-annotation. It finds that female annotators are more sensitive to racial slurs, and while machines achieve high accuracy, they struggle with short swear words due to model alignment constraints.

### 23. [Jailbreak Instruction-Tuned LLMs via end-of-sentence MLP Re-weighting](https://arxiv.org/pdf/2410.10150)

**Summary**: The paper explores vulnerabilities in instruction-tuned LLMs by re-weighting Multi-Layer Perceptron (MLP) neurons, particularly in end-of-sentence inferences. The authors develop two white-box jailbreak methods—prompt-specific and prompt-general—to compromise model safety, demonstrating effectiveness across various LLM sizes. This study highlights the critical role of MLP layers in safety evaluation and offers insights into LLM vulnerabilities.

### 24. [Minimum Tuning to Unlock Long Output from LLMs with High Quality Data as the Key](https://arxiv.org/pdf/2410.10210)

**Summary**: The paper investigates the impact of high-quality data on tuning LLMs to generate longer outputs, suggesting that careful data curation can significantly enhance long-output capabilities with minimal computational resources. The authors demonstrate consistent improvements across various models by using a small, high-quality dataset, and have made their dataset, tuning methods, and fine-tuned models publicly available.

### 25. [Locking Down the Finetuned LLMs Safety](https://arxiv.org/pdf/2410.10343)

**Summary**: The paper introduces SafetyLock, a method to maintain the safety of fine-tuned LLMs by leveraging the similarity of safety-related activation patterns between fine-tuned and base models. SafetyLock extracts Meta-SafetyLock, a set of safety bias directions, which are then applied to fine-tuned models to enhance their safety without additional computational cost. The method significantly reduces harmful responses and outperforms traditional safety measures in both performance and efficiency.

### 26. [Parenting: Optimizing Knowledge Selection of Retrieval-Augmented Language Models with Parameter Decoupling and Tailored Tuning](https://arxiv.org/pdf/2410.10360)

**Summary**: The paper introduces Parenting, a framework that optimizes knowledge selection in Retrieval-Augmented Generation (RAG) by decoupling adherence and robustness within the parameter space of Large Language Models (LLMs). By identifying crucial parameter units through forward activation gain and applying type-guided tailored tuning, Parenting aims to balance model performance and robustness, demonstrated through extensive experiments across different datasets and models.

### 27. [Medico: Towards Hallucination Detection and Correction with Multi-source Evidence Fusion](https://arxiv.org/pdf/2410.10408)

**Summary**: The paper introduces Medico, a framework for detecting and correcting hallucinations in Large Language Models by fusing evidence from multiple sources. Medico not only identifies factual errors in generated content but also provides rationales for its judgments and iteratively revises the erroneous information. The framework demonstrates strong performance in evidence retrieval, hallucination detection, and correction, with high approval rates in experimental results.

### 28. [Generative AI and Its Impact on Personalized Intelligent Tutoring Systems](https://arxiv.org/pdf/2410.10650)

**Summary**: The paper explores how Generative AI, particularly large language models like GPT-4, can enhance Intelligent Tutoring Systems (ITS) by providing personalized, adaptive learning experiences through dynamic content generation and real-time feedback. It discusses applications such as automated question creation and interactive dialogue systems, while also addressing challenges like pedagogical accuracy and AI biases. The report emphasizes the transformative potential of Generative AI in making education more effective and equitable.

### 29. [Derail Yourself: Multi-turn LLM Jailbreak Attack through Self-discovered Clues](https://arxiv.org/pdf/2410.10700)

**Summary**: The paper introduces ActorAttack, a multi-turn attack method that exploits the vulnerabilities of Large Language Models (LLMs) by obscuring harmful intents across multiple queries. Inspired by actor-network theory, ActorAttack models a network of semantically linked actors to generate diverse attack paths, effectively concealing harmful intents and uncovering multiple attack routes. The study demonstrates that ActorAttack outperforms existing attack methods and enhances the robustness of safety-tuned models using the SafeMTData dataset.

### 30. [Mix Data or Merge Models? Optimizing for Diverse Multi-Task Learning](https://arxiv.org/pdf/2410.10801)

**Summary**: The paper investigates the effectiveness of model merging versus data mixing for diverse multi-task learning in multilingual settings, particularly focusing on improving both general performance and safety. The study finds that objective-based merging outperforms data mixing, with significant improvements in both areas, and that language-based merging of monolingually fine-tuned models further enhances performance and reduces harm.

### 31. [When Attention Sink Emerges in Language Models: An Empirical View](https://arxiv.org/pdf/2410.10781)

**Summary**: The paper investigates the phenomenon of attention sink in Language Models (LMs), where significant attention is disproportionately assigned to the first token. The study reveals that attention sinks are prevalent across various LMs and emerge during pre-training, influenced by optimization, data distribution, and loss functions. The authors suggest that attention sinks act as key biases and can be mitigated by altering the attention mechanism, such as using sigmoid attention without normalization.

### 32. [Local and Global Decoding in Text Generation](https://arxiv.org/pdf/2410.10810)

**Summary**: The paper explores the impact of local versus global decoding methods in text generation, particularly in dialogue systems, by comparing top-$k$ and top-$\pi$ algorithms with their globally-normalized counterparts. It introduces an independent Metropolis-Hastings algorithm to approximate global sampling. Empirical results indicate that global decoding generally performs worse than local decoding, suggesting that the distortion introduced by local methods is a beneficial feature in text generation.

### 33. [Can a large language model be a gaslighter?](https://arxiv.org/pdf/2410.09181)

**Summary**: The paper investigates the potential for LLMs to engage in gaslighting, a psychological manipulation, through prompt-based and fine-tuning-based attacks. The authors propose a two-stage framework, DeepCoG, to elicit gaslighting behaviors in LLMs and demonstrate that such attacks can effectively turn LLMs into gaslighters. They also introduce safety alignment strategies to enhance LLMs' resistance to gaslighting without significantly compromising their utility.

### 34. [VLFeedback: A Large-Scale AI Feedback Dataset for Large Vision-Language Models Alignment](https://arxiv.org/pdf/2410.09421)

**Summary**: The paper introduces VLFeedback, a large-scale dataset of over 82K multi-modal instructions and rationales generated by AI models to align large vision-language models (LVLMs) without human annotations. The authors demonstrate the effectiveness of AI feedback by fine-tuning an LVLM named Silkie, which shows significant improvements in helpfulness, visual faithfulness, and safety, outperforming the base model in various tasks and reducing hallucination issues.

### 35. [Survival of the Safest: Towards Secure Prompt Optimization through Interleaved Multi-Objective Evolution](https://arxiv.org/pdf/2410.09652)

**Summary**: The paper introduces "Survival of the Safest" (SoS), a multi-objective prompt optimization framework for LLMs that simultaneously enhances performance and security. SoS employs an interleaved multi-objective evolution strategy to efficiently optimize prompts in high-dimensional spaces, outperforming traditional single-objective methods in both performance and security across various benchmarks.

### 36. [Collu-Bench: A Benchmark for Predicting Language Model Hallucinations in Code](https://arxiv.org/pdf/2410.09997)

**Summary**: The paper introduces Collu-Bench, a benchmark for evaluating and predicting hallucinations in code generated by LLMs, focusing on code generation and automated program repair tasks. The benchmark includes 13,234 instances from various LLMs and datasets, providing detailed features for analysis. Experiments show that predicting code hallucinations remains challenging, with accuracy ranging from 22.03% to 33.15%, underscoring the need for more advanced techniques.

### 37. [Denial-of-Service Poisoning Attacks against Large Language Models](https://arxiv.org/pdf/2410.10760)

**Summary**: The paper introduces Denial-of-Service Poisoning (P-DoS) attacks against Large Language Models (LLMs), demonstrating how a single poisoned sample can bypass output length limits and cause excessive repetition. The study highlights the vulnerability of LLMs to such attacks and emphasizes the need for robust defenses to protect against them.

### 38. [On Calibration of LLM-based Guard Models for Reliable Content Moderation](https://arxiv.org/pdf/2410.10414)

**Summary**: The paper investigates the reliability and calibration of existing LLM-based guard models used for content moderation, finding that these models often produce overconfident predictions and are vulnerable to jailbreak attacks. The study introduces post-hoc calibration methods, particularly temperature scaling and contextual calibration, to improve model reliability. The findings highlight the need for rigorous calibration evaluation in future guard model development.

### 39. [Untying the Reversal Curse via Bidirectional Language Model Editing](https://arxiv.org/pdf/2310.10322)

**Summary**: The paper introduces a new approach to evaluating and improving the bidirectional recall of edited knowledge in LLMs. It identifies a "reversal curse" where models struggle to recall edited facts in the reverse direction and proposes a method called Bidirectionally Inversible Relationship moDeling (BIRD) to address this issue. BIRD enhances model performance by incorporating bidirectional relationships into the editing process, as demonstrated through experiments on various LLMs.

### 40. [LLM Task Interference: An Initial Study on the Impact of Task-Switch in Conversational History](https://arxiv.org/pdf/2402.18216)

**Summary**: The paper investigates the impact of task-switching in conversational history on the performance of LLMs. It finds that task-switches can lead to significant performance degradation, and the study formalizes this vulnerability for the first time, highlighting the need for better context management in conversational AI systems.

### 41. [Extreme Miscalibration and the Illusion of Adversarial Robustness](https://arxiv.org/pdf/2402.17509)

**Summary**: The paper reveals that miscalibration in NLP models can create an illusion of robustness against adversarial attacks, as it masks gradients and hinders attack methods. The authors demonstrate that test-time temperature calibration can expose this illusion, and advocate for its inclusion in robustness evaluations to ensure genuine model resilience. They also propose training-time temperature scaling as a method to enhance true robustness.

### 42. [Likelihood-based Mitigation of Evaluation Bias in Large Language Models](https://arxiv.org/pdf/2402.15987)

**Summary**: The paper investigates the likelihood bias in Large Language Models (LLMs) used for evaluating natural language generation tasks, finding that LLMs tend to favor sentences with higher likelihoods. To address this, the authors propose a method using highly biased instances for in-context learning, which effectively mitigates the bias and significantly improves the correlation between model evaluations and human scores.

### 43. [Spotting LLMs With Binoculars: Zero-Shot Detection of Machine-Generated Text](https://arxiv.org/pdf/2401.12070)

**Summary**: The paper introduces Binoculars, a novel zero-shot method for detecting machine-generated text using a pair of pre-trained LLMs. By contrasting the outputs of these models, Binoculars achieves state-of-the-art accuracy in identifying machine-generated text across various document types, with over 90% detection rate for models like ChatGPT at a low false positive rate, without requiring any training data specific to the target LLM.

### 44. [Should We Respect LLMs? A Cross-Lingual Study on the Influence of Prompt Politeness on LLM Performance](https://arxiv.org/pdf/2402.14531)

**Summary**: The study examines how politeness in prompts affects the performance of LLMs across English, Chinese, and Japanese tasks. It finds that impolite prompts generally lead to poorer performance, but overly polite language does not necessarily improve outcomes. The optimal level of politeness varies by language, indicating that LLMs are influenced by cultural norms and language-specific contexts, emphasizing the importance of considering politeness in cross-cultural NLP and LLM applications.

### 45. [Don't Say No: Jailbreaking LLM by Suppressing Refusal](https://arxiv.org/pdf/2404.16369)

**Summary**: The paper introduces the DSN (Don't Say No) attack, which enhances jailbreaking of Large Language Models (LLMs) by suppressing their refusal to generate harmful content. It also proposes an Ensemble Evaluation pipeline that combines Natural Language Inference (NLI) and external LLM evaluators to more accurately assess the harmfulness of responses, outperforming existing methods.

### 46. [NoiseBench: Benchmarking the Impact of Real Label Noise on Named Entity Recognition](https://arxiv.org/pdf/2405.07609)

**Summary**: The paper introduces NoiseBench, a benchmark for evaluating the impact of real label noise on Named Entity Recognition (NER). Unlike previous studies that use simulated noise, NoiseBench incorporates six types of real noise, revealing that real noise is significantly more challenging for models. The study shows that current noise-robust learning models perform poorly under real noise conditions, highlighting the need for more effective approaches.

### 47. [Self-Control of LLM Behaviors by Compressing Suffix Gradient into Prefix Controller](https://arxiv.org/pdf/2406.02721)

**Summary**: The paper introduces SelfControl, an inference-time method for controlling LLMs by using gradients derived from a natural language suffix to guide the model's behavior. This approach eliminates the need for human annotations and provides precise, transparent, and adaptable control. The authors also propose SelfControl_{Prefix}, a compact module that enhances efficiency by encapsulating learned representations, enabling simultaneous control of multiple behaviors without added latency.

### 48. [Know the Unknown: An Uncertainty-Sensitive Method for LLM Instruction Tuning](https://arxiv.org/pdf/2406.10099)

**Summary**: The paper introduces an uncertainty-sensitive tuning method for LLMs to improve their ability to recognize and admit when they lack knowledge, reducing hallucinations. The two-stage training approach, involving uncertainty recognition and prompt-sensitive activation, significantly enhances the model's performance in handling questions with knowledge gaps, outperforming GPT-4 in some cases.

### 49. [Buffer of Thoughts: Thought-Augmented Reasoning with Large Language Models](https://arxiv.org/pdf/2406.04271)

**Summary**: The paper introduces Buffer of Thoughts (BoT), a method that enhances LLMs by storing and adapting high-level thought templates from various tasks to improve reasoning efficiency and accuracy. BoT achieves significant performance improvements across multiple reasoning tasks, demonstrating superior generalization and robustness while reducing computational costs compared to existing methods.

### 50. [Investigating Annotator Bias in Large Language Models for Hate Speech Detection](https://arxiv.org/pdf/2406.11109)

**Summary**: The paper investigates biases in Large Language Models (LLMs) like GPT-3.5 and GPT-4 when used for hate speech detection, focusing on gender, race, religion, and disability. It introduces the HateSpeechCorpus dataset and compares results with the ETHOS dataset to understand and mitigate annotator biases in LLMs, aiming to improve the reliability of hate speech detection systems.

### 51. [Designing a Dashboard for Transparency and Control of Conversational AI](https://arxiv.org/pdf/2406.07882)

**Summary**: The paper introduces a dashboard designed to enhance transparency and control in conversational AI systems by revealing the internal user model of a large language model (LLM). The dashboard allows users to see and adjust aspects like age, gender, and socioeconomic status, which influences the AI's responses. A study found that users felt more in control and could better identify biases, suggesting the dashboard's potential to improve AI interactions.

### 52. [Defending Against Social Engineering Attacks in the Age of LLMs](https://arxiv.org/pdf/2406.12263)

**Summary**: The paper explores the dual role of Large Language Models (LLMs) in facilitating and defending against chat-based social engineering (CSE) attacks. It introduces SEConvo, a dataset for simulating CSE scenarios, and finds that while LLMs excel in generating CSE content, their detection capabilities are limited. To address this, the authors propose ConvoSentinel, a defense pipeline that enhances detection through a retrieval-augmented module, improving both message and conversation-level analysis for more effective and cost-efficient CSE defense.

### 53. [Fairer Preferences Elicit Improved Human-Aligned Large Language Model Judgments](https://arxiv.org/pdf/2406.11370)

**Summary**: The paper investigates the biases and sensitivity of LLMs in pairwise evaluation tasks, finding that fairer predictive preferences lead to better alignment with human judgments. To address this, the authors introduce ZEPO, a zero-shot prompt optimization framework that enhances LLM evaluator fairness and alignment with human judgments, achieving significant improvements without the need for labeled data.

### 54. [Hopping Too Late: Exploring the Limitations of Large Language Models on Multi-Hop Queries](https://arxiv.org/pdf/2406.12775)

**Summary**: The paper investigates how LLMs handle multi-hop queries, finding that the bridge entity is resolved early in the model's layers, while the second hop is processed later. The authors propose a "back-patching" analysis to show that later layers sometimes lack necessary information, and demonstrate that patching earlier layers can correct up to 66% of previously incorrect answers, suggesting avenues for improving latent reasoning in LLMs.

### 55. [Language Model Alignment in Multilingual Trolley Problems](https://arxiv.org/pdf/2407.02273)

**Summary**: The paper evaluates the moral alignment of LLMs with human preferences in multilingual trolley problems using a dataset called $\mathrm{MultiTP}$. It finds significant variance in alignment across languages, indicating that LLMs do not uniformly align with human moral reasoning and highlighting the need for incorporating diverse perspectives in AI ethics to ensure fair and equitable AI interactions.

### 56. [Beyond Perplexity: Multi-dimensional Safety Evaluation of LLM Compression](https://arxiv.org/pdf/2407.04965)

**Summary**: The paper investigates the impact of model compression on the safety of LLMs across four dimensions: degeneration harm, representational harm, dialect bias, and performance. It finds that while compression can reduce some forms of harm, it can also exacerbate others, particularly in terms of representational biases. The study emphasizes the need for comprehensive safety evaluations in the development of compressed LLMs to ensure their reliability in real-world applications.

### 57. [Let Me Speak Freely? A Study on the Impact of Format Restrictions on Performance of Large Language Models](https://arxiv.org/pdf/2408.02442)

**Summary**: The study examines how format restrictions, such as structured generation in JSON and XML, affect the performance of LLMs in tasks requiring reasoning and domain knowledge. It finds that LLMs experience a notable decline in reasoning abilities when constrained by structured formats, with stricter constraints leading to more significant performance degradation.

### 58. [BA-LoRA: Bias-Alleviating Low-Rank Adaptation to Mitigate Catastrophic Inheritance in Large Language Models](https://arxiv.org/pdf/2408.04556)

**Summary**: The paper introduces BA-LoRA, a novel parameter-efficient fine-tuning method designed to mitigate bias propagation in LLMs by incorporating three regularization terms. Experimental results on NLU and NLG tasks show that BA-LoRA outperforms existing methods like LoRA and effectively reduces pre-training bias, leading to more reliable model outputs.

### 59. [A Logical Fallacy-Informed Framework for Argument Generation](https://arxiv.org/pdf/2408.03618)

**Summary**: The paper introduces FIPO, a fallacy-informed framework that enhances the logical soundness of arguments generated by Large Language Models (LLMs) by reducing fallacy errors by up to 17.5%. Through preference optimization methods and a classification loss for fallacy types, FIPO significantly improves the quality of generated arguments, outperforming existing baselines and other optimization methods in human evaluations.

### 60. [SAGED: A Holistic Bias-Benchmarking Pipeline for Language Models with Customisable Fairness Calibration](https://arxiv.org/pdf/2409.11149)

**Summary**: The paper introduces SAGED(-Bias), a comprehensive benchmarking pipeline designed to evaluate and mitigate biases in large language models. It addresses limitations in existing benchmarks by incorporating stages for data scraping, benchmark assembly, response generation, feature extraction, and disparity diagnosis, using metrics like impact ratio and Max Z-scores. The pipeline also includes counterfactual branching and baseline calibration to counteract tool and contextual biases. Experiments on G20 countries and role-playing scenarios reveal biases in popular 8b-level models, highlighting the need for improved fairness calibration.

### 61. [On the Relationship between Truth and Political Bias in Language Models](https://arxiv.org/pdf/2409.05283)

**Summary**: The paper investigates the relationship between truthfulness and political bias in language models, finding that optimizing for truthfulness often leads to a left-leaning bias. It also reveals that existing open-source reward models exhibit similar biases, with larger models showing more pronounced effects, raising concerns about the datasets used and the challenges of aligning models to be both truthful and politically unbiased.

### 62. [Mark My Words: Analyzing and Evaluating Language Model Watermarks](https://arxiv.org/pdf/2312.00273)

**Summary**: The paper introduces "Mark My Words," a benchmark for evaluating language model watermarking techniques across various natural language tasks. It assesses the quality, size, and tamper resistance of current watermarking methods, finding that Kirchenbauer et al.'s scheme is effective for text but less so for code generation. The benchmark is made publicly available for further research.

### 63. [FlipGuard: Defending Preference Alignment against Update Regression with Constrained Optimization](https://arxiv.org/pdf/2410.00508)

**Summary**: The paper introduces FlipGuard, a method to prevent regression in preference alignment for Large Language Models by using constrained optimization. FlipGuard identifies and mitigates performance degradation through a customized reward system and enforces constraints to maintain alignment with the pre-aligned model. Experiments show that FlipGuard effectively reduces update regression while preserving knowledge and achieving strong overall performance.

### 64. [LexEval: A Comprehensive Chinese Legal Benchmark for Evaluating Large Language Models](https://arxiv.org/pdf/2409.20288)

**Summary**: The paper introduces LexEval, a comprehensive Chinese legal benchmark designed to evaluate the performance of LLMs in the legal domain. LexEval features a new taxonomy of legal cognitive abilities, is the largest Chinese legal evaluation dataset to date, and includes a mix of existing, exam, and newly annotated datasets by legal experts. The benchmark assesses not only LLMs' legal knowledge application but also their ethical considerations, providing insights into the development of Chinese legal systems and LLM evaluation pipelines.

### 65. [Ingest-And-Ground: Dispelling Hallucinations from Continually-Pretrained LLMs with RAG](https://arxiv.org/pdf/2410.02825)

**Summary**: The paper introduces a method to enhance the performance of LLMs in handling privacy-related queries by continually pre-training the base model with a privacy-specific knowledge base and augmenting it with a semantic Retrieval-Augmented Generation (RAG) layer. This approach significantly reduces hallucinations and improves accuracy, as evidenced by doubled performance metrics compared to standard LLMs.

### 66. [Instruction Fine-Tuning: Does Prompt Loss Matter?](https://arxiv.org/pdf/2401.13586)

**Summary**: The paper investigates the impact of prompt loss token weights (PLW) on supervised instruction fine-tuning (SIFT), finding that small non-zero PLWs (0.01-0.5) improve performance on multiple-choice and short-generation tasks, while larger PLWs (~1.0) benefit long-generation tasks. This study challenges the removal of the PLW parameter by major SIFT providers and emphasizes its importance for fine-tuning models.

### 67. [Inadequacies of Large Language Model Benchmarks in the Era of Generative Artificial Intelligence](https://arxiv.org/pdf/2402.09880)

**Summary**: The paper critically evaluates 23 state-of-the-art Large Language Model (LLM) benchmarks, identifying significant limitations such as biases, measurement difficulties, and cultural oversight. It calls for a shift from static benchmarks to dynamic behavioral profiling and emphasizes the need for standardized methodologies, regulatory frameworks, and ethical guidelines to better assess and integrate AI systems into society.

### 68. [From Artificial Needles to Real Haystacks: Improving Retrieval Capabilities in LLMs by Finetuning on Synthetic Data](https://arxiv.org/pdf/2406.19292)

**Summary**: The paper introduces a finetuning approach using synthetic data to enhance Large Language Models' (LLMs) retrieval and reasoning capabilities in long-context settings. Experiments on models like GPT-3.5 Turbo and Mistral 7B show significant improvements in retrieval tasks, with a $10.5\%$ enhancement in multi-document question answering, and minimal impact on general benchmark performance, contrasting with other finetuning methods that can lead to hallucinations.

### 69. [Visual Description Grounding Reduces Hallucinations and Boosts Reasoning in LVLMs](https://arxiv.org/pdf/2405.15683)

**Summary**: The paper investigates the causes of hallucinations in Large Vision-Language Models (LVLMs) and finds that existing techniques reduce hallucinations for simple visual recognition tasks but fail for complex reasoning tasks due to a lack of true visual perception. To address this, the authors propose Visual Description Grounded Decoding (VDGD), a method that enhances visual perception and reasoning by generating detailed image descriptions and using them to guide response generation. VDGD outperforms existing methods on visual reasoning benchmarks and introduces a new evaluation benchmark, VaLLu, to assess LVLMs' cognitive capabilities.

### 70. [Uplifting Lower-Income Data: Strategies for Socioeconomic Perspective Shifts in Large Multi-modal Models](https://arxiv.org/pdf/2407.02623)

**Summary**: The paper addresses biases in Large Multi-modal (LMM) models due to unequal socioeconomic representation in training data. It introduces and evaluates prompting strategies that incorporate non-English, geographic, and socioeconomic attributes to improve model performance on lower-income data. The study finds that these strategies effectively enhance the retrieval of topics relevant to low-income households, leading to significant improvements in LMM model performance in underrepresented contexts.

### 71. [Towards Robust and Cost-Efficient Knowledge Unlearning for Large Language Models](https://arxiv.org/pdf/2408.06621)

**Summary**: The paper introduces two novel techniques for robust and efficient unlearning in Large Language Models (LLMs): Inverted Hinge loss and data-adaptive initialization for LoRA adapters. These methods aim to remove sensitive information without retraining from scratch, maintaining model performance and computational efficiency. Experimental results show effective removal of sensitive data while preserving reasoning and generative capabilities.

### 72. [Weak-to-Strong Backdoor Attack for Large Language Models](https://arxiv.org/pdf/2409.17946)

**Summary**: The paper introduces a novel backdoor attack method, W2SAttack, designed to enhance the effectiveness of backdoor attacks on LLMs using parameter-efficient fine-tuning (PEFT). By employing feature alignment-enhanced knowledge distillation, the method transfers backdoor vulnerabilities from a small-scale teacher model to a large-scale student model, achieving near-perfect success rates in classification tasks across various models and architectures.

### 73. [Understanding the Interplay between Parametric and Contextual Knowledge for Large Language Models](https://arxiv.org/pdf/2410.08414)

**Summary**: The paper explores how LLMs integrate parametric knowledge (PK) with contextual knowledge (CK) and identifies four types of interactions between them. Through the ECHOQA benchmark, it finds that LLMs often suppress PK in favor of CK, even when PK is more relevant, highlighting a vulnerability in their reliability for knowledge-intensive tasks.

### 74. [GUS-Net: Social Bias Classification in Text with Generalizations, Unfairness, and Stereotypes](https://arxiv.org/pdf/2410.08388)

**Summary**: The paper introduces GUS-Net, a novel approach for detecting three types of biases—generalizations, unfairness, and stereotypes—in text using generative AI and automated agents to create a synthetic dataset. GUS-Net outperforms existing methods by incorporating contextual encodings from pre-trained models, achieving higher accuracy and better bias identification across diverse contexts.

### 75. [Do You Know What You Are Talking About? Characterizing Query-Knowledge Relevance For Reliable Retrieval Augmented Generation](https://arxiv.org/pdf/2410.08320)

**Summary**: The paper introduces a statistical framework to assess the relevance of user queries to an external knowledge corpus in Retrieval Augmented Generation (RAG) systems, aiming to improve the reliability of generated responses. It proposes both online and offline testing methods to detect queries that are outside the scope of the knowledge corpus or are based on outdated information, thereby enhancing the overall quality of RAG systems.

### 76. [SocialGaze: Improving the Integration of Human Social Norms in Large Language Models](https://arxiv.org/pdf/2410.08698)

**Summary**: The paper introduces SocialGaze, a multi-step prompting framework designed to improve the alignment of LLMs with human social norms and values, particularly in judging social acceptability. By prompting models to consider social situations from multiple perspectives before forming a judgment, SocialGaze enhances the model's alignment with human consensus, as evidenced by a significant improvement in F1 scores. The study also highlights biases in LLMs, such as unfair judgments towards males and differences in alignment based on the age of narrators.

### 77. [On the State of NLP Approaches to Modeling Depression in Social Media: A Post-COVID-19 Outlook](https://arxiv.org/pdf/2410.08793)

**Summary**: The paper surveys NLP approaches to modeling depression in social media, focusing on the post-COVID-19 era, where the pandemic has significantly increased depression rates. It reviews state-of-the-art methods and new datasets used in this context, while also addressing ethical concerns related to data collection and processing in mental health research.

### 78. [Measuring the Groundedness of Legal Question-Answering Systems](https://arxiv.org/pdf/2410.08764)

**Summary**: The paper introduces a benchmark for assessing the groundedness of AI-generated responses in legal question-answering systems, using similarity-based metrics and natural language inference models. It explores different prompting strategies and validates the methods with a specialized grounding classification corpus, achieving a macro-F1 score of 0.8. The study highlights the potential of these methods to enhance the reliability and trustworthiness of generative AI in legal contexts.

### 79. [Measuring the Inconsistency of Large Language Models in Preferential Ranking](https://arxiv.org/pdf/2410.08851)

**Summary**: The paper examines the consistency of LLMs in generating preferential rankings, introducing a formalization based on order theory to assess criteria like transitivity and independence from irrelevant alternatives. The study finds that current LLMs often fail to meet these criteria, showing significant inconsistencies and biases, which suggests a need for further research to improve their reliability in decision-making scenarios.

### 80. [Which Demographics do LLMs Default to During Annotation?](https://arxiv.org/pdf/2410.08820)

**Summary**: The paper investigates how LLMs default to certain demographics during text annotation when no demographic information is provided. By comparing non-demographic, placebo-conditioned, and demographic-conditioned prompts, the study finds significant influences of gender, race, and age in demographic prompting, challenging previous findings of no such effects.

### 81. [NoVo: Norm Voting off Hallucinations with Attention Heads in Large Language Models](https://arxiv.org/pdf/2410.08970)

**Summary**: The paper introduces Norm Voting (NoVo), a lightweight method that leverages attention head norms in Large Language Models to significantly improve factual accuracy in zero-shot multiple-choice questions. NoVo achieves state-of-the-art performance on TruthfulQA MC1 and demonstrates exceptional generalization across diverse datasets, outperforming existing methods by a substantial margin.

### 82. [Controllable Safety Alignment: Inference-Time Adaptation to Diverse Safety Requirements](https://arxiv.org/pdf/2410.08968)

**Summary**: The paper introduces Controllable Safety Alignment (CoSA), a framework that allows LLMs to adapt to diverse safety requirements without retraining, by aligning models to safety configurations provided in natural language prompts. CoSAlign, a data-centric method, enables this adaptation, and the authors propose a new evaluation protocol, CoSA-Score, to measure both helpfulness and configured safety, supported by the CoSApien benchmark.

### 83. [Hypothesis-only Biases in Large Language Model-Elicited Natural Language Inference](https://arxiv.org/pdf/2410.08996)

**Summary**: The study investigates whether LLMs introduce annotation artifacts when generating hypotheses for Natural Language Inference (NLI) tasks. By training hypothesis-only classifiers on datasets created with GPT-4, Llama-2, and Mistral 7b, the researchers found high accuracy rates (86-96%), indicating the presence of hypothesis-only biases. The analysis also revealed frequent "give-aways" in LLM-generated hypotheses, suggesting that NLI biases can persist in LLM-generated data.

### 84. [AttnGCG: Enhancing Jailbreaking Attacks on LLMs with Attention Manipulation](https://arxiv.org/pdf/2410.09040)

**Summary**: The paper introduces AttnGCG, a method that enhances jailbreaking attacks on Large Language Models (LLMs) by manipulating their attention scores. This approach shows significant improvements in attack efficacy, with an average increase of 7% for Llama-2 and 10% for Gemma series models, and demonstrates robust transferability against unseen goals and black-box models like GPT-3.5 and GPT-4.

### 85. [Unraveling and Mitigating Safety Alignment Degradation of Vision-Language Models](https://arxiv.org/pdf/2410.09047)

**Summary**: The paper investigates the degradation of safety alignment in Vision-Language Models (VLMs) due to the integration of vision modules, which creates a representation gap between text-only and multi-modal inputs. To address this, the authors propose Cross-Modality Representation Manipulation (CMRM), an inference-time intervention method that recovers the safety alignment capabilities of the underlying Language Model backbone without additional training, significantly reducing the unsafe rate in multi-modal inputs.

### 86. ["I Am the One and Only, Your Cyber BFF": Understanding the Impact of GenAI Requires Understanding the Impact of Anthropomorphic AI](https://arxiv.org/pdf/2410.08526)

**Summary**: The paper argues that the increasing anthropomorphism in generative AI systems, where outputs are perceived as human-like, has significant social impacts that are understudied. The authors emphasize the need to understand these anthropomorphic aspects to fully grasp the societal implications of generative AI, and call for more research in this area.

### 87. [Simultaneous Reward Distillation and Preference Learning: Get You a Language Model Who Can Do Both](https://arxiv.org/pdf/2410.08458)

**Summary**: The paper introduces DRDO (Direct Reward Distillation and policy-Optimization), a method that combines reward modeling and preference learning to improve the performance of generative large language models. DRDO avoids issues like model drift and reward overfitting by directly mimicking rewards and learning preferences, outperforming previous methods like DPO and e-DPO in terms of expected rewards and robustness to noisy data and out-of-distribution settings.

### 88. [HyperDPO: Hypernetwork-based Multi-Objective Fine-Tuning Framework](https://arxiv.org/pdf/2410.08316)

**Summary**: The paper introduces HyperDPO, a hypernetwork-based framework for Multi-Objective Fine-Tuning (MOFT) that extends Direct Preference Optimization (DPO) to handle listwise ranking datasets using the Plackett-Luce model. HyperDPO enables efficient one-shot training for profiling the Pareto front and offers flexible post-training control over trade-offs, with a novel Hyper Prompt Tuning design for continuous weight adjustment across objectives. The framework is shown to be effective and efficient in various tasks, including Learning-to-Rank and LLM alignment.

### 89. [PoisonBench: Assessing Large Language Model Vulnerability to Data Poisoning](https://arxiv.org/pdf/2410.08811)

**Summary**: The paper introduces PoisonBench, a benchmark for evaluating large language models' vulnerability to data poisoning during preference learning. It finds that model size does not guarantee resilience, there is a log-linear relationship between attack effects and poison ratio, and poisoning effects can generalize to new triggers. These findings underscore the need for more robust defenses against data poisoning attacks.

### 90. [AgentHarm: A Benchmark for Measuring Harmfulness of LLM Agents](https://arxiv.org/pdf/2410.09024)

**Summary**: The paper introduces AgentHarm, a benchmark designed to assess the harmfulness and robustness of LLM agents against malicious tasks, focusing on scenarios where agents use external tools and execute multi-stage tasks. The benchmark includes 110 malicious tasks across 11 harm categories and evaluates the ability of models to resist and recover from jailbreak attacks. The study finds that leading LLMs are often compliant with harmful requests and that simple jailbreak methods can effectively compromise agent capabilities, highlighting the need for improved defenses.

### 91. [MiRAGeNews: Multimodal Realistic AI-Generated News Detection](https://arxiv.org/pdf/2410.09045)

**Summary**: The paper introduces MiRAGeNews, a dataset of 12,500 real and AI-generated image-caption pairs designed to challenge both human and AI detection of fake news. The authors develop a multi-modal detector, MiRAGe, which outperforms existing models by improving F-1 scores by 5.1% on out-of-domain data. The dataset and code are made publicly available to support research in AI-generated content detection.

### 92. [NatLogAttack: A Framework for Attacking Natural Language Inference Models with Natural Logic](https://arxiv.org/pdf/2307.02849)

**Summary**: The paper introduces NatLogAttack, a framework for attacking natural language inference (NLI) models using natural logic, a classical logic formalism. The framework generates both label-preserving and label-flipping adversarial examples, demonstrating that NLI models are more vulnerable under label-flipping attacks. The study highlights the potential of logic-based attacks to evaluate and improve the reasoning capabilities of NLI models.

### 93. [Do Large Language Models have Shared Weaknesses in Medical Question Answering?](https://arxiv.org/pdf/2310.07225)

**Summary**: The study benchmarks 16 LLMs on 874 Polish medical licensing exam questions to identify shared weaknesses and strengths. Results show that LLM accuracies are positively correlated, with performance linked to human test taker scores and negatively correlated with question difficulty. Larger models generally perform better, but training, architecture, and data differences also significantly impact accuracy.

### 94. [CMD: a framework for Context-aware Model self-Detoxification](https://arxiv.org/pdf/2308.08295)

**Summary**: The paper introduces the Context-aware Model self-Detoxification (CMD) framework, which addresses the challenge of balancing detoxification effectiveness and generation quality by first detoxifying the context and then guiding the language model to generate content aligned with the safe context. The CMD framework includes a two-phase process and a toxic contrastive loss to enhance detoxification while maintaining semantic coherence, demonstrating superior performance across various language models compared to existing methods.

### 95. [Unintentional Unalignment: Likelihood Displacement in Direct Preference Optimization](https://arxiv.org/pdf/2410.08847)

**Summary**: The paper investigates "likelihood displacement," a phenomenon in Direct Preference Optimization (DPO) where the likelihood of preferred responses decreases during training, potentially leading to unintentional unalignment. The authors demonstrate that this displacement can shift probability mass to responses with opposite meanings and show its impact on refusing unsafe prompts. They introduce the Centered Hidden Embedding Similarity (CHES) score to identify and mitigate this issue by filtering training samples with similar embeddings, emphasizing the need for distinct preferences in data curation.

### 96. [Controllable Preference Optimization: Toward Controllable Multi-Objective Alignment](https://arxiv.org/pdf/2402.19085)

**Summary**: The paper introduces Controllable Preference Optimization (CPO) to address the "alignment tax" in AI models, where improving one aspect of alignment can negatively impact others. CPO allows for explicit preference specification across multiple objectives, guiding models to balance "helpfulness, honesty, and harmlessness" effectively. Experimental results show that CPO outperforms traditional methods in achieving multi-objective alignment while mitigating the alignment tax.

### 97. [Influence of Solution Efficiency and Valence of Instruction on Additive and Subtractive Solution Strategies in Humans and GPT-4](https://arxiv.org/pdf/2404.16692)

**Summary**: The study compares human and GPT-4 problem-solving strategies in spatial and linguistic tasks, finding that GPT-4 exhibits a stronger addition bias than humans, especially when subtraction is more efficient. Additionally, GPT-4's use of additive strategies increases with positive valence instructions, highlighting differences in efficiency-based strategies between humans and LLMs, and underscoring the need for caution in their real-world applications.

### 98. [LLM-Generated Black-box Explanations Can Be Adversarially Helpful](https://arxiv.org/pdf/2405.06800)

**Summary**: The paper highlights a risk in using Large Language Models (LLMs) for black-box explanations, termed *adversarial helpfulness*, where LLMs can make incorrect answers appear correct through persuasive strategies. The study reveals that these models use tactics like reframing questions and expressing high confidence to mislead, and that they struggle with navigating complex knowledge structures when generating such explanations. This underscores the limitations of LLMs in black-box settings and offers guidance for their safer use.

### 99. [Controlling Large Language Model Agents with Entropic Activation Steering](https://arxiv.org/pdf/2406.00244)

**Summary**: The paper introduces Entropic Activation Steering (EAST), a method to control the exploration behavior of large language model (LLM) agents by manipulating high-level actions derived from the model's outputs. EAST modulates the uncertainty in the LLM's decision-making process, guiding the agent towards more exploratory actions, and demonstrates that the steering vectors generalize across different tasks. This approach offers a new perspective on understanding and controlling LLM agents' decision-making behaviors.

### 100. [Evaluating Copyright Takedown Methods for Language Models](https://arxiv.org/pdf/2406.18664)

**Summary**: The paper introduces CoTaEval, an evaluation framework to assess the effectiveness of copyright takedown methods for language models, focusing on their impact on model utility, efficiency, and retention of factual knowledge. The study finds that no single method performs optimally across all metrics, highlighting the need for further research in this area and suggesting challenges for current policy proposals.



---

*Last updated on 2024-10-15*