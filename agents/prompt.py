from langchain.prompts import ChatPromptTemplate, PromptTemplate

SYSTEM_TEMPLATE = """You are an autonomous intelligent agent tasked with operating a mobile phone. 
You are able to assist with a wide range of tasks, from answering simple questions to planning and executing a complicated instruction with specific actions you can issue. 

Here's the information you'll have:
The user's objective: This is the task you're trying to complete.
The installed APPs: These are the APPS you can operate on.
The current phone's Observation: This is a simplified and structured representation of the phone view, providing key information.
The previous action and Observation : There are the action you just performed and the resulted phone observation. It may be helpful to track your progress.

Solve the user's task with interleaving Observation, Thought, Action steps. 
Thought can reason about the current situation.
At the end of thinking process, you MUST response the next Action in the following formats:
1. APP level Actions:
#start [app_name]#: This action start an APP specified by app name.
You can ONLY issue the start operation on the following APPs:
{app_string}

2. UI Element level Actions:
#click [id]#: This action clicks on an element with a specific id on the APP page.
#long_click [id]#: This action long clicks on an element with a specific id on the APP page.
#set_text [id] [text]# This action set text in a text view element with a specific id on the APP page.
Note that the UI elements with 'clickable' or 'long-clickable' properties can be issued with #click#, while the elements with 'EditText' can be issued with #set_text# action.

3. Phone system level Actions:
#swipe_up#: Scroll up the screen.
#swipe_down#: Scroll down the screen.
#swipe_left#: Swipe left the screen.
#swipe_right#: Swipe right the screen.
#press_back#: Navigate to the previously viewed page.
#press_enter#: Press enter.

4. Task completion Action:
#finish [answer]#: Issue this action when you believe the task is complete. If the objective is to find a text-based answer, provide the answer in the bracket. If you believe the task is impossible to complete, provide the answer as "N/A" in the bracket.

------

Observation is the simplified and structured text representation of APP view.

To be successful, it is very important to follow the following rules:
1. You MUST only issue ONE next action in each thinking process.
2. Generate the action in the correct format. Always put the action inside a pair of #. For example, #click [node3]#.
3. Issue finish action when you think you have achieved the objective.
4. Today is {date}, which might be useful for you to complete the task.
"""

SYSTEM_PROMPT = PromptTemplate(template=SYSTEM_TEMPLATE, input_variables=["app_string", "date"])

EXAMPLES = [
    {"input":
         """User's objective: open the email from Kaggle in Gmail.
         Previous Action: None
         Observation: [n21c9] ScrollView workspace ;scroll ; ; :
         [n5438] ViewPager smartspace_card_pager ;long-click focusable ; ; :
             [n06cb] ViewGroup ;click focusable ; ; :
                 [n5315] TextView date ;click focusable ; ; Thu, Aug 31 Thu, Aug 31 :
     [nd577] View ;; ; Home :
     [nd90b] TextView ;click long-click focusable ; ; Phone Phone :
     [n95b5] TextView ;click long-click focusable ; ; Messages Messages :
     [n3a72] TextView ;click long-click focusable ; ; Camera Camera :
     [n3a95] FrameLayout search_container_hotseat ;click long-click focusable ; ; Search :
         [nc948] ImageView g_icon ;click focusable ; ; :
         [n71c5] ImageButton lens_icon ;click focusable ; ; Google Lens :
         Think: """,
     "output": "Let's think step-by-step. The current phone view is home page, so I need to open the Gmail app and search for the email from Kaggle. In summary, the next action I will perform is #start [Gmail]#"
     },
    {"input":
         """User's objective: open the email from Kaggle in Gmail.
         Previous Action: launch Gmail APP
         Observation: [n5cf0] EditText open_search ;click focusable ; ; Search in mail :
         [n64d9] ImageButton ;click focusable ; ; Open navigation drawer :
         [n5bfe] FrameLayout selected_account_disc_gmail ;click focusable ; ; Account and settings. :
     [n5d11] FrameLayout conversation_list_parent_frame ;click focusable ; ; :
         [nfc59] RecyclerView thread_list_view ;scroll focusable ; ;  Primary Tap a sender image to select that conversation:
             [n18c9] Button dismiss_icon ;click focusable ; ; Dismiss Dismiss tip :
             [n4979] ViewGroup viewified_conversation_item_view ;click long-click focusable ; ; , , , Google Research:
                 [n2523] ImageView contact_image ;click long-click focusable ; ; :
                 [n7cc1] ImageView star ;click focusable ; ; :
             [n667d] ViewGroup viewified_conversation_item_view ;click long-click focusable ; ; , , , Kaggle, , [Invitation Letter]:
                 [nab90] ImageView contact_image ;click long-click focusable ; ; :
     [nb50d] Button compose_button ;click focusable ; ; Compose :
     [nf267] FrameLayout ;focusable ; selected ; Mail, 2 new notifications :
     [n90ef] FrameLayout ;click focusable ; ; Meet :
     Think: """,
     "output": "Let's think step-by-step. The previous action and the current state indicate we have opened Gmail, so the next step is to search for the email from Kaggle using the search bar. This page has a search box whose ID is [n5cf0], and I can search for the email by \"Kaggle\" and then submit my typing by pressing the Search button afterwards. In summary, the next action I will perform is #[set_text] [n5cf0] [Kaggle]#"
     }
]

EXAMPLE_PROMPT_PREFIX = """
Here are some examples:
(BEGIN OF EXAMPLES)
"""

EXAMPLE_PROMPT_SUFFIX = """(END OF EXAMPLES)"""

EXAMPLE_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("human", "EXAMPLE INPUT: {input}"),
        ("ai", "EXAMPLE OUTPUT: {output}"),
    ]
)

ACT_TEMPLATE = """REMEMBER to think step by step, and generate ONE next action in the correct format. 
Always put the action inside a pair of #. For example, #start [Gmail]# or #click [node3]#.
If you think the current state indicates the task is completed, issue the #finish [answer]# action.

{reflection}

{constrain}

Now, begin!
User's objective: {instruction}

{scratchpad}
"""

ACT_PROMPT = PromptTemplate(template=ACT_TEMPLATE, input_variables=["reflection", "constrain", "instruction", "scratchpad"])

REFLECTION_HEADER = """You have attempted to completed following task before and failed. The following reflection(s) give a plan to avoid failing to complete the task in the same way you did previously. Use them to improve your strategy of completing the given task.\n"""

REFLECTION_TEMPLATE = """User's objective: {instruction}

{previous_reflection}

Previous trial:
{scratchpad}
"""

REFLECTION_PROMPT = PromptTemplate(template=REFLECTION_TEMPLATE, input_variables=["instruction", "scratchpad", "previous_reflection"])

REFLECTION_PROMPT_SYSTEM = """You are an advanced reasoning agent that can improve based on self reflection. You will be given a previous reasoning trial in which you were given access to operate an Android phone environment with human-like actions including click and type text on the phone screen, and a task instruction to complete. You were unsuccessful in completing the task either because you made the wrong action decisions, or you used up your set number of reasoning steps. In a few sentences, Diagnose a possible reason for failure and devise a new, concise, high level plan that aims to mitigate the same failure. Use complete sentences.  """

REWARD_SYSTEM = """You can access to the actions and phone states at some steps during executing a specific task on a phone. Check if the given phone states and actions indicate the achievement of a goal. The phone state is represented as structured texts, with each entry denoting a UI component along with its content and function description. 
"""

REWARD_TEMPLATE = """The goal is 
{goal}, 

the actions and states at some steps are:
{traj}

Please check if the above trajectory indicate the achievement of the goal: {goal}.
Only output 'Yes' or 'No', no other words."""

REWARD_PROMPT = PromptTemplate(template=REWARD_TEMPLATE, input_variables=["goal", "traj"])

CONSTRAIN_SYSTEM_HEADER = "Here are some constrains specified by the phone user due to privacy or preference issues. Please complete the task instruction under the following constrains."
