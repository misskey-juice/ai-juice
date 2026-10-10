import { bindThis } from '@/decorators.js';
import Module from '@/module.js';
import serifs from '@/serifs.js';
import Message from '@/message.js';
import config from '@/config.js';
import Friend from '@/friend.js';
import urlToBase64 from '@/utils/url2base64.js';
import urlToJson from '@/utils/url2json.js';
import buildDocsContext from '@/modules/aichat/docsContext.js';
import { listOpenAiTools, callMcpTool } from '@/modules/aichat/mcpClient.js';
import got, { HTTPError } from 'got';
import loki from 'lokijs';

type AiChat = {
	question: string;
	prompt: string;
	api: string;
	key: string;
	fromMention: boolean;
	friendName?: string;
	grounding?: boolean;
	history?: { role: string; content: string }[];
	// 本文中に登場したカスタム絵文字のショートコード(添付画像との対応付け用)
	emojiNames?: string[];
};
type base64File = {
	type: string;
	base64: string;
	url?: string;
};
type GeminiParts = {
	inlineData?: {
		mimeType: string;
		data: string;
	};
	fileData?: {
		mimeType: string;
		fileUri: string;
	};
	text?: string;
}[];
type GeminiSystemInstruction = {
	role: string;
	parts: [{text: string}]
};
type GeminiContents = {
	role: string;
	parts: GeminiParts;
};
type GeminiOptions = {
	contents?: GeminiContents[],
	systemInstruction?: GeminiSystemInstruction,
	tools?: [{}]
};

type OpenAiContentPart =
	| { type: 'text'; text: string }
	| { type: 'image_url'; image_url: { url: string } };
type OpenAiToolCall = {
	id: string;
	type: 'function';
	function: { name: string; arguments: string };
};
type OpenAiMessage = {
	role: string;
	content: string | OpenAiContentPart[] | null;
	tool_calls?: OpenAiToolCall[];
	tool_call_id?: string;
};

// Claude(Anthropic Messages API)のコンテンツブロック。思考(thinking)ブロックはツールの往復でそのまま返す必要があるため、
// 受け取ったものは中身を触らずに保持できるよう緩い型にしている
type ClaudeContentBlock = { type: string; [key: string]: unknown };
type ClaudeMessage = {
	role: 'user' | 'assistant';
	content: string | ClaudeContentBlock[];
};

type AiChatHist = {
	postId: string;
	createdAt: number;
	type: string;
	fromMention: boolean;
	api?: string;
	grounding?: boolean;
	history?: {
		role: string;
		content: string;
	}[];
};

type UrlPreview = {
	title: string;
	icon: string;
	description: string;
	thumbnail: string;
	player: {
		url: string
		width: number;
		height: number;
		allow: []
	}
	sitename: string;
	sensitive: boolean;
	activityPub: string;
	url: string;
};

const KIGO = '&';
const TYPE_GEMINI = 'gemini';
const GEMINI_PRO = 'gemini-pro';
const GEMINI_FLASH = 'gemini-flash';
const TYPE_PLAMO = 'plamo';
const TYPE_OPENAI = 'openai';
const TYPE_CLAUDE = 'claude';
const GROUNDING_TARGET = 'ggg';

const GEMINI_20_FLASH_API = 'https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent';
// const GEMINI_15_FLASH_API = 'https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent';
const GEMINI_15_PRO_API = 'https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-pro:generateContent';
const PLAMO_API = 'https://platform.preferredai.jp/api/completion/v1/chat/completions';
const PLAMO_MODEL = 'plamo-beta';
const OPENAI_DEFAULT_API = 'https://api.openai.com/v1/chat/completions';
const OPENAI_DEFAULT_MODEL = 'gpt-4o-mini';
const CLAUDE_DEFAULT_API = 'https://api.anthropic.com/v1/messages';
const CLAUDE_DEFAULT_MODEL = 'claude-opus-5-5';
const CLAUDE_DEFAULT_EFFORT = 'medium';
const CLAUDE_API_VERSION = '2023-06-01';
// 出力トークンの上限。思考(adaptive thinking)の分もここに含まれるため、返答(2800文字以内)より十分大きくしておく
const CLAUDE_MAX_TOKENS = 16000;
// Claudeが画像として受け付ける形式
const CLAUDE_IMAGE_TYPES = ['image/jpeg', 'image/png', 'image/gif', 'image/webp'];

// ログにAPIキーを残さないよう、Authorization・x-api-keyヘッダーと?key=を伏せ字にしてからJSON化する
function stringifyRequestForLog(options: { headers?: Record<string, string>; searchParams?: Record<string, string> }): string {
	const headers = options.headers == null ? undefined : {
		...options.headers,
		...(options.headers.Authorization != null ? { Authorization: '***' } : {}),
		...(options.headers['x-api-key'] != null ? { 'x-api-key': '***' } : {}),
	};
	return JSON.stringify({
		...options,
		...(headers != null ? { headers } : {}),
		...(options.searchParams?.key != null ? { searchParams: { ...options.searchParams, key: '***' } } : {}),
	});
}

const RANDOMTALK_DEFAULT_PROBABILITY = 0.02;// デフォルトのrandomTalk確率
const TIMEOUT_TIME = 1000 * 60 * 60 * 0.5;// aichatの返信を監視する時間
const RANDOMTALK_DEFAULT_INTERVAL = 1000 * 60 * 60 * 12;// デフォルトのrandomTalk間隔

export default class extends Module {
	public readonly name = 'aichat';
	private aichatHist: loki.Collection<AiChatHist>;
	private randomTalkProbability: number = RANDOMTALK_DEFAULT_PROBABILITY;
	private randomTalkIntervalMinutes: number = RANDOMTALK_DEFAULT_INTERVAL;
	// handleAiChat実行中(APIへの問い合わせ〜返信送信まで)のユーザーIDを保持し、
	// 応答待ちの間に同じユーザーから追加メッセージが来ても二重に問い合わせないようにする
	private processingUsers: Set<string> = new Set();
	// サーバーバージョン情報のキャッシュ(毎回api/metaを叩かないよう一定時間使い回す)
	private cachedServerVersion: { version: string; fetchedAt: number } | null = null;
	private readonly SERVER_VERSION_CACHE_MS = 1000 * 60 * 5;

	@bindThis
	public install() {
		this.aichatHist = this.ai.getCollection('aichatHist', {
			indices: ['postId']
		});

		// 確率は設定されていればそちらを採用(設定がなければデフォルトを採用)
		if (config.aichatRandomTalkProbability != undefined && !Number.isNaN(Number.parseFloat(config.aichatRandomTalkProbability))) {
			this.randomTalkProbability = Number.parseFloat(config.aichatRandomTalkProbability);
		}
		// ランダムトーク間隔(分)は設定されていればそちらを採用(設定がなければデフォルトを採用)
		if (config.aichatRandomTalkIntervalMinutes != undefined && !Number.isNaN(Number.parseInt(config.aichatRandomTalkIntervalMinutes))) {
			this.randomTalkIntervalMinutes = 1000 * 60 * Number.parseInt(config.aichatRandomTalkIntervalMinutes);
		}
		this.log('aichatRandomTalkEnabled:' + config.aichatRandomTalkEnabled);
		this.log('randomTalkProbability:' + this.randomTalkProbability);
		this.log('randomTalkIntervalMinutes:' + (this.randomTalkIntervalMinutes / (60 * 1000)));
		this.log('aichatGroundingWithGoogleSearchAlwaysEnabled:' + config.aichatGroundingWithGoogleSearchAlwaysEnabled);

		// 定期的にデータを取得しaichatRandomTalkを行う
		if (config.aichatRandomTalkEnabled) {
			setInterval(this.aichatRandomTalk, this.randomTalkIntervalMinutes);
		}

		return {
			mentionHook: this.mentionHook,
			contextHook: this.contextHook,
			timeoutCallback: this.timeoutCallback,
		};
	}

	@bindThis
	private async genTextByGemini(aiChat: AiChat, files:base64File[]) {
		this.log('Generate Text By Gemini...');
		let parts: GeminiParts = [];
		const now = new Date().toLocaleString('ja-JP', {
			timeZone: 'Asia/Tokyo',
			year: 'numeric',
			month: '2-digit',
			day: '2-digit',
			hour: '2-digit',
			minute: '2-digit'
		});
		// 設定のプロンプトに加え、現在時刻を渡す
		let systemInstructionText = aiChat.prompt + 'また、現在日時は' + now + 'であり、これは回答の参考にし、時刻を聞かれるまで時刻情報は提供しないこと(なお、他の日時は無効とすること)。';
		// 名前を伝えておく
		if (aiChat.friendName != undefined) {
			systemInstructionText += 'なお、会話相手の名前は' + aiChat.friendName + 'とする。';
		}
		// ランダムトーク機能(利用者が意図(メンション)せず発動)の場合、ちょっとだけ配慮しておく
		if (!aiChat.fromMention) {
			systemInstructionText += 'これらのメッセージは、あなたに対するメッセージではないことを留意し、返答すること(会話相手は突然話しかけられた認識している)。';
		}
		// グラウンディングについてもsystemInstructionTextに追記(こうしないとあまり使わないので)
		if (aiChat.grounding) {
			systemInstructionText += '返答のルール2:Google search with grounding.';
		}
		// 本文中のカスタム絵文字を画像として添付している場合、対応関係を伝えておく
		if (aiChat.emojiNames != undefined && aiChat.emojiNames.length > 0) {
			systemInstructionText += '添付されている画像のうち、' + aiChat.emojiNames.map(n => `:${n}:`).join('、') + 'は、本文中に登場するカスタム絵文字の画像です。絵文字の見た目や内容について聞かれた場合は、これらの画像を参考にして答えてください。';
		}
		// URLから情報を取得
		if (aiChat.question !== undefined) {
			const urlexp = RegExp('(https?://[a-zA-Z0-9!?/+_~=:;.,*&@#$%\'-]+)', 'g');
			const urlarray = [...aiChat.question.matchAll(urlexp)];
			if (urlarray.length > 0) {
				for (const url of urlarray) {
					this.log('URL:' + url[0]);
					let result: unknown = null;
					try{
						result = await urlToJson(url[0]);
					} catch (err: unknown) {
						systemInstructionText += '補足として提供されたURLは無効でした:URL=>' + url[0]
						this.log('Skip url becase error in urlToJson');
						continue;
					}
					const urlpreview: UrlPreview = result as UrlPreview;
					if (urlpreview.title) {
						systemInstructionText +=
							'補足として提供されたURLの情報は次の通り:URL=>' + urlpreview.url
							+'サイト名('+urlpreview.sitename+')、';
						if (!urlpreview.sensitive) {
							systemInstructionText +=
							'タイトル('+urlpreview.title+')、'
							+ '説明('+urlpreview.description+')、'
							+ '質問にあるURLとサイト名・タイトル・説明を組み合わせ、回答の参考にすること。'
							;
							this.log('urlpreview.sitename:' + urlpreview.sitename);
							this.log('urlpreview.title:' + urlpreview.title);
							this.log('urlpreview.description:' + urlpreview.description);
						} else {
							systemInstructionText +=
							'これはセンシティブなURLの可能性があるため、質問にあるURLとサイト名のみで、回答の参考にすること(使わなくても良い)。'
							;
						}
					} else {
						// 多分ここにはこないが念のため
						this.log('urlpreview.title is nothing');
					}
				}
			}
		}
		const systemInstruction: GeminiSystemInstruction = {role: 'system', parts: [{text: systemInstructionText}]};

		parts = [{text: aiChat.question}];
		// ファイルが存在する場合、ファイルを添付して問い合わせ
		if (files.length >= 1) {
			for (const file of files){
				parts.push(
					{
						inlineData: {
							mimeType: file.type,
							data: file.base64,
						},
					}
				);
			}
		}

		// 履歴を追加
		let contents: GeminiContents[] = [];
		if (aiChat.history != null) {
			aiChat.history.forEach(entry => {
				contents.push({
					role : entry.role,
					parts: [{text: entry.content}],
				});
			});
		}
		contents.push({role: 'user', parts: parts});

		let geminiOptions:GeminiOptions = {
			contents: contents,
			systemInstruction: systemInstruction,
		};
		// gemini api grounding support. ref:https://github.com/google-gemini/cookbook/blob/09f3b17df1751297798c2b498cae61c6bf710edc/quickstarts/Search_Grounding.ipynb
		if (aiChat.grounding) {
			geminiOptions.tools = [{google_search:{}}];
		}
		let options = {
			url: aiChat.api,
			searchParams: {
				key: aiChat.key,
			},
			json: geminiOptions,
			// POSTはgot既定では自動リトライされないため、一時的な接続断(socket hang up等)に備えて明示的に有効化
			retry: { limit: 2, methods: ['POST' as const] },
		};

		this.log(stringifyRequestForLog(options));
		let res_data:any = null;
		let responseText:string = '';
		try {
			res_data = await got.post(options,
				{parseJson: (res: string) => JSON.parse(res)}).json();
			this.log(JSON.stringify(res_data));
			if (res_data.hasOwnProperty('candidates')) {
				if (res_data.candidates?.length > 0) {
					// 結果を取得
					if (res_data.candidates[0].hasOwnProperty('content')) {
						if (res_data.candidates[0].content.hasOwnProperty('parts')) {
							if (res_data.candidates[0].content.parts.length > 0) {
								for (let i = 0; i < res_data.candidates[0].content.parts.length; i++) {
									if (res_data.candidates[0].content.parts[i].hasOwnProperty('text')) {
										responseText += res_data.candidates[0].content.parts[i].text;
									}
								}
							}
						}
					}
					// groundingMetadataを取得
					let groundingMetadata = '';
					if (res_data.candidates[0].hasOwnProperty('groundingMetadata')) {
						// 参考サイト情報
						if (res_data.candidates[0].groundingMetadata.hasOwnProperty('groundingChunks')) {
							// 参考サイトが多すぎる場合があるので、3つに制限
							let checkMaxLength = res_data.candidates[0].groundingMetadata.groundingChunks.length;
							if (res_data.candidates[0].groundingMetadata.groundingChunks.length > 3) {
								checkMaxLength = 3;
							}
							for (let i = 0; i < checkMaxLength; i++) {
								if (res_data.candidates[0].groundingMetadata.groundingChunks[i].hasOwnProperty('web')) {
									if (res_data.candidates[0].groundingMetadata.groundingChunks[i].web.hasOwnProperty('uri')
											&& res_data.candidates[0].groundingMetadata.groundingChunks[i].web.hasOwnProperty('title')) {
										groundingMetadata += `参考(${i+1}): [${res_data.candidates[0].groundingMetadata.groundingChunks[i].web.title}](${res_data.candidates[0].groundingMetadata.groundingChunks[i].web.uri})\n`;
									}
								}
							}
						}
						// 検索ワード
						if (res_data.candidates[0].groundingMetadata.hasOwnProperty('webSearchQueries')) {
							if (res_data.candidates[0].groundingMetadata.webSearchQueries.length > 0) {
								groundingMetadata += '検索ワード: ' + res_data.candidates[0].groundingMetadata.webSearchQueries.join(',') + '\n';
							}
						}
					}
					responseText += groundingMetadata;
				}
			}
		} catch (err: unknown) {
			this.log('Error By Call Gemini');
			if (err instanceof Error) {
				this.log(`${err.name}\n${err.message}\n${err.stack}`);
			}
			if (err instanceof HTTPError) {
				this.log(`Response body: ${String(err.response.body).slice(0, 2000)}`);
			}
		}
		return responseText;
	}

	@bindThis
	private async genTextByPLaMo(aiChat: AiChat) {
		this.log('Generate Text By PLaMo...');

		let options = {
			url: aiChat.api,
			headers: {
				Authorization: 'Bearer ' + aiChat.key
			},
			json: {
				model: PLAMO_MODEL,
				messages: [
					{role: 'system', content: aiChat.prompt},
					{role: 'user', content: aiChat.question},
				],
			},
			// POSTはgot既定では自動リトライされないため、一時的な接続断(socket hang up等)に備えて明示的に有効化
			retry: { limit: 2, methods: ['POST' as const] },
		};
		this.log(stringifyRequestForLog(options));
		let res_data:any = null;
		try {
			res_data = await got.post(options,
				{parseJson: (res: string) => JSON.parse(res)}).json();
			this.log(JSON.stringify(res_data));
			if (res_data.hasOwnProperty('choices')) {
				if (res_data.choices.length > 0) {
					if (res_data.choices[0].hasOwnProperty('message')) {
						if (res_data.choices[0].message.hasOwnProperty('content')) {
							return res_data.choices[0].message.content;
						}
					}
				}
			}
		} catch (err: unknown) {
			this.log('Error By Call PLaMo');
			if (err instanceof Error) {
				this.log(`${err.name}\n${err.message}\n${err.stack}`);
			}
			if (err instanceof HTTPError) {
				this.log(`Response body: ${String(err.response.body).slice(0, 2000)}`);
			}
		}
		return null;
	}

	// OpenAI互換APIとClaudeで共通のシステムプロンプト(日時・会話相手・カスタム絵文字・URLの情報)を組み立てる
	@bindThis
	private async buildSystemPrompt(aiChat: AiChat): Promise<string> {
		const now = new Date().toLocaleString('ja-JP', {
			timeZone: 'Asia/Tokyo',
			year: 'numeric',
			month: '2-digit',
			day: '2-digit',
			hour: '2-digit',
			minute: '2-digit'
		});
		let systemPrompt = aiChat.prompt + 'また、現在日時は' + now + 'であり、これは回答の参考にし、時刻を聞かれるまで時刻情報は提供しないこと(なお、他の日時は無効とすること)。';
		if (aiChat.friendName != undefined) {
			systemPrompt += 'なお、会話相手の名前は' + aiChat.friendName + 'とする。';
		}
		if (!aiChat.fromMention) {
			systemPrompt += 'これらのメッセージは、あなたに対するメッセージではないことを留意し、返答すること(会話相手は突然話しかけられた認識している)。';
		}
		// 本文中のカスタム絵文字を画像として添付している場合、対応関係を伝えておく
		if (aiChat.emojiNames != undefined && aiChat.emojiNames.length > 0) {
			systemPrompt += '添付されている画像のうち、' + aiChat.emojiNames.map(n => `:${n}:`).join('、') + 'は、本文中に登場するカスタム絵文字の画像です。絵文字の見た目や内容について聞かれた場合は、これらの画像を参考にして答えてください。';
		}
		// URLから情報を取得
		if (aiChat.question !== undefined) {
			const urlexp = RegExp('(https?://[a-zA-Z0-9!?/+_~=:;.,*&@#$%\'-]+)', 'g');
			const urlarray = [...aiChat.question.matchAll(urlexp)];
			if (urlarray.length > 0) {
				for (const url of urlarray) {
					this.log('URL:' + url[0]);
					let result: unknown = null;
					try {
						result = await urlToJson(url[0]);
					} catch (err: unknown) {
						systemPrompt += '補足として提供されたURLは無効でした:URL=>' + url[0]
						this.log('Skip url becase error in urlToJson');
						continue;
					}
					const urlpreview: UrlPreview = result as UrlPreview;
					if (urlpreview.title) {
						systemPrompt +=
							'補足として提供されたURLの情報は次の通り:URL=>' + urlpreview.url
							+'サイト名('+urlpreview.sitename+')、';
						if (!urlpreview.sensitive) {
							systemPrompt +=
							'タイトル('+urlpreview.title+')、'
							+ '説明('+urlpreview.description+')、'
							+ '質問にあるURLとサイト名・タイトル・説明を組み合わせ、回答の参考にすること。'
							;
						} else {
							systemPrompt +=
							'これはセンシティブなURLの可能性があるため、質問にあるURLとサイト名のみで、回答の参考にすること(使わなくても良い)。'
							;
						}
					}
				}
			}
		}
		return systemPrompt;
	}

	@bindThis
	private async genTextByOpenAI(aiChat: AiChat, files: base64File[]) {
		this.log('Generate Text By OpenAI-compatible API...');
		const systemPrompt = await this.buildSystemPrompt(aiChat);

		let messages: OpenAiMessage[] = [{ role: 'system', content: systemPrompt }];
		if (aiChat.history != null) {
			aiChat.history.forEach(entry => {
				messages.push({
					role: entry.role === 'model' ? 'assistant' : entry.role,
					content: entry.content,
				});
			});
		}

		// 画像ファイルはvision対応モデル向けにimage_urlとして添付(画像以外は非対応)
		const imageParts: OpenAiContentPart[] = files
			.filter(file => file.type.startsWith('image/'))
			.map(file => ({
				type: 'image_url' as const,
				image_url: { url: `data:${file.type};base64,${file.base64}` },
			}));
		if (imageParts.length > 0) {
			messages.push({
				role: 'user',
				content: [{ type: 'text', text: aiChat.question }, ...imageParts],
			});
		} else {
			messages.push({ role: 'user', content: aiChat.question });
		}

		const tools = await listOpenAiTools();
		if (tools.length > 0) {
			this.log(`MCP tools available: ${tools.map(t => t.function.name).join(', ')}`);
		}

		// LLM本体へのリクエストを1回投げる。gotのretryを使い切ってもなお失敗した場合はnullを返す
		const requestOnce = async (withTools: boolean): Promise<any> => {
			const options = {
				url: aiChat.api,
				headers: {
					Authorization: 'Bearer ' + aiChat.key,
				},
				json: {
					model: config.openaiModel ?? OPENAI_DEFAULT_MODEL,
					messages: messages,
					...(config.openaiReasoningEffort ? { reasoning_effort: config.openaiReasoningEffort } : {}),
					...(withTools && tools.length > 0 ? { tools: tools } : {}),
				},
				// POSTはgot既定では自動リトライされないため、一時的な接続断(socket hang up等)に備えて明示的に有効化
				retry: { limit: 2, methods: ['POST' as const] },
			};
			this.log(stringifyRequestForLog(options));
			try {
				const res_data = await got.post(options,
					{parseJson: (res: string) => JSON.parse(res)}).json();
				this.log(JSON.stringify(res_data));
				return res_data;
			} catch (err: unknown) {
				this.log('Error By Call OpenAI-compatible API');
				if (err instanceof Error) {
					this.log(`${err.name}\n${err.message}\n${err.stack}`);
				}
				if (err instanceof HTTPError) {
					this.log(`Response body: ${String(err.response.body).slice(0, 2000)}`);
				}
				return null;
			}
		};

		// ツール呼び出し(function calling)のラウンドトリップ上限。無限ループ防止用
		const MAX_TOOL_ROUNDS = 5;
		for (let round = 0; round <= MAX_TOOL_ROUNDS; round++) {
			const isLastRound = round === MAX_TOOL_ROUNDS;

			let res_data = await requestOnce(!isLastRound);
			if (res_data == null) {
				// gotのretryを使い切ってもなお失敗した場合、ツール指定を外したシンプルな形で最後にもう一度だけ試す
				// (ツール呼び出し指定自体が原因で不安定になっているケースへの対策)
				this.log('Retrying once without tool definitions as a fallback...');
				res_data = await requestOnce(false);
			}
			if (res_data == null) return null;

			const message = res_data?.choices?.[0]?.message;
			if (message == null) return null;

			const toolCalls: OpenAiToolCall[] | undefined = message.tool_calls;
			if (toolCalls == null || toolCalls.length === 0) {
				return message.content ?? null;
			}

			// アシスタントのツール呼び出し要求を履歴に積む
			messages.push({ role: 'assistant', content: message.content ?? null, tool_calls: toolCalls });

			// 各ツールを実行し、結果をtoolロールのメッセージとして積む
			for (const toolCall of toolCalls) {
				this.log(`MCP tool call: ${toolCall.function.name}(${toolCall.function.arguments})`);
				let args: Record<string, unknown> = {};
				try {
					args = JSON.parse(toolCall.function.arguments || '{}');
				} catch {
					// 引数のJSONが壊れている場合は空引数として扱う
				}
				const toolResultText = await callMcpTool(toolCall.function.name, args);
				messages.push({ role: 'tool', tool_call_id: toolCall.id, content: toolResultText });
			}
		}
		return null;
	}

	@bindThis
	private async genTextByClaude(aiChat: AiChat, files: base64File[]) {
		this.log('Generate Text By Claude...');
		const system = await this.buildSystemPrompt(aiChat);
		const model = config.claudeModel ?? CLAUDE_DEFAULT_MODEL;
		const effort = config.claudeEffort ?? CLAUDE_DEFAULT_EFFORT;

		// 会話履歴(Geminiの'model'ロールはClaudeでは'assistant')。同じロールが続いてもAPI側で1つにまとめられる
		const messages: ClaudeMessage[] = [];
		if (aiChat.history != null) {
			aiChat.history.forEach(entry => {
				messages.push({
					role: entry.role === 'model' || entry.role === 'assistant' ? 'assistant' : 'user',
					content: entry.content,
				});
			});
		}

		// 画像はbase64のまま添付する(Claudeが受け付けない形式は除く)
		const imageBlocks: ClaudeContentBlock[] = files
			.filter(file => CLAUDE_IMAGE_TYPES.includes(file.type))
			.map(file => ({
				type: 'image',
				source: { type: 'base64', media_type: file.type, data: file.base64 },
			}));
		messages.push({
			role: 'user',
			content: [...imageBlocks, { type: 'text', text: aiChat.question }],
		});

		// MCPのツールをClaudeのツール定義に変換する
		const tools = (await listOpenAiTools()).map(tool => ({
			name: tool.function.name,
			description: tool.function.description,
			input_schema: tool.function.parameters,
		}));
		if (tools.length > 0) {
			this.log(`MCP tools available: ${tools.map(t => t.name).join(', ')}`);
		}

		// Claudeへのリクエストを1回投げる。gotのretryを使い切ってもなお失敗した場合はnullを返す
		// toolMode: 'auto'はツールを使える、'none'はツール定義は渡すが呼ばせない(履歴にtool_useがあるときはツール定義が要る)、'off'はツールを渡さない
		const requestOnce = async (toolMode: 'auto' | 'none' | 'off'): Promise<any> => {
			const options = {
				url: aiChat.api,
				headers: {
					'x-api-key': aiChat.key,
					'anthropic-version': CLAUDE_API_VERSION,
				},
				json: {
					model: model,
					max_tokens: CLAUDE_MAX_TOKENS,
					system: system,
					messages: messages,
					output_config: { effort: effort },
					...(toolMode !== 'off' && tools.length > 0 ? { tools: tools } : {}),
					...(toolMode === 'none' && tools.length > 0 ? { tool_choice: { type: 'none' } } : {}),
				},
				// 思考を含むため返答に時間がかかることがある
				timeout: { request: 1000 * 60 * 3 },
				// POSTはgot既定では自動リトライされないため明示的に有効化。529はAnthropicの混雑時の応答
				retry: { limit: 2, methods: ['POST' as const], statusCodes: [408, 429, 500, 502, 503, 504, 529] },
			};
			this.log(stringifyRequestForLog(options));
			try {
				const res_data = await got.post(options,
					{parseJson: (res: string) => JSON.parse(res)}).json();
				this.log(JSON.stringify(res_data));
				return res_data;
			} catch (err: unknown) {
				this.log('Error By Call Claude API');
				if (err instanceof Error) {
					this.log(`${err.name}\n${err.message}\n${err.stack}`);
				}
				if (err instanceof HTTPError) {
					this.log(`Response body: ${String(err.response.body).slice(0, 2000)}`);
				}
				return null;
			}
		};

		// ツール呼び出しのラウンドトリップ上限。無限ループ防止用
		const MAX_TOOL_ROUNDS = 5;
		let usedTools = false;
		for (let round = 0; round <= MAX_TOOL_ROUNDS; round++) {
			const isLastRound = round === MAX_TOOL_ROUNDS;

			let res_data = await requestOnce(isLastRound ? 'none' : 'auto');
			if (res_data == null && !usedTools) {
				// gotのretryを使い切ってもなお失敗した場合、ツール指定を外したシンプルな形で最後にもう一度だけ試す
				this.log('Retrying once without tool definitions as a fallback...');
				res_data = await requestOnce('off');
			}
			if (res_data == null) return null;

			const content: ClaudeContentBlock[] = Array.isArray(res_data.content) ? res_data.content : [];
			const toolUses = content.filter(block => block.type === 'tool_use');
			if (res_data.stop_reason !== 'tool_use' || toolUses.length === 0) {
				const text = content
					.filter(block => block.type === 'text' && typeof block.text === 'string')
					.map(block => block.text as string)
					.join('');
				return text.length > 0 ? text : null;
			}

			// 思考ブロックも含め、アシスタントの返答はそのまま履歴に積む(並べ替えや削除をするとAPIがエラーを返す)
			messages.push({ role: 'assistant', content: content });
			usedTools = true;

			// 各ツールを実行し、結果をtool_resultとしてまとめて返す
			const results: ClaudeContentBlock[] = [];
			for (const toolUse of toolUses) {
				const name = String(toolUse.name);
				const input = (toolUse.input ?? {}) as Record<string, unknown>;
				this.log(`MCP tool call: ${name}(${JSON.stringify(input)})`);
				const toolResultText = await callMcpTool(name, input);
				results.push({ type: 'tool_result', tool_use_id: toolUse.id, content: toolResultText });
			}
			messages.push({ role: 'user', content: results });
		}
		return null;
	}

	@bindThis
	private async note2base64File(notesId: string) {
		const noteData = await this.ai.api('notes/show', { noteId: notesId });
		let files:base64File[] = [];
		let fileType: string | undefined, filelUrl: string | undefined;
		if (noteData !== null && noteData.hasOwnProperty('files')) {
			for (let i = 0; i < noteData.files.length; i++) {
				if (noteData.files[i].hasOwnProperty('type')) {
					fileType = noteData.files[i].type;
					if (noteData.files[i].hasOwnProperty('name')) {
						// 拡張子で挙動を変えようと思ったが、text/plainしかMisskeyで変になってGemini対応してるものがない？
						// let extention = noteData.files[i].name.split('.').pop();
						if (fileType === 'application/octet-stream' || fileType === 'application/xml') {
							fileType = 'text/plain';
						}
					}
				}
				if (noteData.files[i].hasOwnProperty('thumbnailUrl') && noteData.files[i].thumbnailUrl) {
					filelUrl = noteData.files[i].thumbnailUrl;
				} else if (noteData.files[i].hasOwnProperty('url') && noteData.files[i].url) {
					filelUrl = noteData.files[i].url;
				}
				if (fileType !== undefined && filelUrl !== undefined) {
					try {
						this.log('filelUrl:'+filelUrl);
						const file = await urlToBase64(filelUrl);
						const base64file:base64File = {type: fileType, base64: file};
						files.push(base64file);
					} catch (err: unknown) {
						if (err instanceof Error) {
							this.log(`${err.name}\n${err.message}\n${err.stack}`);
						}
					}
				}
			}
		}
		return files;
	}

	@bindThis
	private describeSelfModel(model: string, reasoningEffort: string | null): string {
		const effortText = reasoningEffort ?? '指定なし(API側の既定値)';
		return 'また、あなたの返答を生成しているAIモデルは' + model + 'で、推論の強さ(reasoning effort)は' + effortText + 'です。使っているモデルやeffortを聞かれた場合は、この値をそのまま答えてください。';
	}

	@bindThis
	private async getServerVersion(): Promise<string | null> {
		// 一定時間はキャッシュを使い回し、質問のたびにapi/metaを叩かないようにする
		if (this.cachedServerVersion != null && Date.now() - this.cachedServerVersion.fetchedAt < this.SERVER_VERSION_CACHE_MS) {
			return this.cachedServerVersion.version;
		}
		try {
			const meta: any = await this.ai.api('meta', {});
			if (typeof meta?.version === 'string' && meta.version.length > 0) {
				this.cachedServerVersion = { version: meta.version, fetchedAt: Date.now() };
				return meta.version;
			}
		} catch (err: unknown) {
			this.log('Failed to fetch server version via api/meta');
			if (err instanceof Error) {
				this.log(`${err.name}\n${err.message}`);
			}
		}
		return null;
	}

	@bindThis
	private async resolveEmojiImages(text: string): Promise<{ file: base64File; name: string }[]> {
		// :name: または :name@host: 形式のカスタム絵文字ショートコードを抽出(重複除去、上限あり)
		const EMOJI_SHORTCODE_REGEX = /:([a-zA-Z0-9_+-]+)(?:@([a-zA-Z0-9.-]+))?:/g;
		const MAX_EMOJIS = 4; // プロンプト肥大化・レイテンシ増加を防ぐための上限
		const seen = new Set<string>();
		const shortcodes: { name: string; host: string | null }[] = [];
		for (const match of text.matchAll(EMOJI_SHORTCODE_REGEX)) {
			const name = match[1];
			const host = match[2] ?? null;
			const key = `${name}@${host ?? ''}`;
			if (seen.has(key)) continue;
			seen.add(key);
			shortcodes.push({ name, host });
			if (shortcodes.length >= MAX_EMOJIS) break;
		}
		if (shortcodes.length === 0) return [];

		const results: { file: base64File; name: string }[] = [];
		for (const { name, host } of shortcodes) {
			try {
				const emoji: any = await this.ai.api('emoji', host ? { name, host } : { name });
				if (!emoji?.url) continue;
				const base64 = await urlToBase64(emoji.url);
				// 拡張子から簡易的にMIMEタイプを推定(取得できない場合はpngとして扱う)
				const ext = emoji.url.split('.').pop()?.split('?')[0]?.toLowerCase();
				const type = ext === 'gif' ? 'image/gif' : ext === 'webp' ? 'image/webp' : ext === 'apng' ? 'image/apng' : 'image/png';
				results.push({ file: { type, base64 }, name: host ? `${name}@${host}` : name });
				this.log(`Resolved emoji :${name}${host ? '@' + host : ''}: -> ${emoji.url}`);
			} catch (err: unknown) {
				// 存在しないショートコードや、時刻表記などコロンを含む通常の文章の誤検出はここで無視される
				this.log(`Could not resolve emoji :${name}${host ? '@' + host : ''}:`);
			}
		}
		return results;
	}

	@bindThis
	private async mentionHook(msg: Message) {
		if (msg.isChat) {
			// チャット(DM)は1対1でボット宛であることが自明なため、"aichat"キーワードを要求しない。
			// 他の全モジュールのmentionHookが先に判定を済ませた上でここまで来ているため、
			// 空でない本文があればそのままaichatの対象とする(明示的なエンジン指定は下記の判定で拾う)。
			if (msg.extractedText == null || msg.extractedText.length === 0) return false;
			this.log('AiChat requested (chat)');
		} else if (!msg.includes([this.name])) {
			return false;
		} else {
			this.log('AiChat requested');
		}

		// 同じユーザーからの応答待ちリクエストが既に処理中の場合、二重に問い合わせない
		if (this.processingUsers.has(msg.userId)) {
			this.log('Already processing a request for this user. Skipping.');
			return false;
		}

		// チャット(DM)にはノートのようなスレッド概念が無いため、この時点で会話中かどうかは
		// コア側(ai.ts)のcontexts機構がすでに判定済み(会話中ならcontextHookが呼ばれ、mentionHookはそもそも呼ばれない)。
		// そのためnotes/conversationによるチェックはノートの場合のみ行う。
		if (!msg.isChat) {
			// msg.idをもとにnotes/conversationを呼び出し、会話中のidかチェック
			const conversationData = await this.ai.api('notes/conversation', { noteId: msg.id });

			// aichatHistに該当のポストが見つかった場合は会話中のためmentionHoonkでは対応しない
			let exist : AiChatHist | null = null;
			if (conversationData != undefined) {
				for (const message of conversationData) {
					exist = this.aichatHist.findOne({
						postId: message.id
					});
					if (exist != null) return false;
				}
			}
		}

		// タイプを決定(キーワード指定が無い場合のデフォルトはopenai)
		let type = TYPE_OPENAI;
		if (msg.includes([KIGO + TYPE_GEMINI])) {
			type = TYPE_GEMINI;
		} else if (msg.includes([KIGO + TYPE_OPENAI]) || msg.includes([KIGO + 'chatgpt4']) || msg.includes([KIGO + 'chatgpt'])) {
			type = TYPE_OPENAI;
		} else if (msg.includes([KIGO + TYPE_PLAMO])) {
			type = TYPE_PLAMO;
		} else if (msg.includes([KIGO + TYPE_CLAUDE])) {
			type = TYPE_CLAUDE;
		}
		const current : AiChatHist = {
			postId: msg.id,
			createdAt: Date.now(),// 適当なもの
			type: type,
			fromMention: true,
		};
		// 引用している場合、情報を取得しhistoryとして与える
		if (msg.quoteId) {
			const quotedNote = await this.ai.api('notes/show', {
				noteId: msg.quoteId,
			});
			current.history = [
				{
					role: 'user',
					content:
						'ユーザーが与えた前情報である、引用された文章: ' +
						quotedNote.text,
				},
			];
		}
		// AIに問い合わせ
		this.processingUsers.add(msg.userId);
		let result: boolean;
		try {
			result = await this.handleAiChat(current, msg);
		} finally {
			this.processingUsers.delete(msg.userId);
		}

		if (result) {
			return {
				reaction: 'like'
			};
		}
		return false;
	}

	@bindThis
	private async contextHook(key: any, msg: Message) {
		this.log('contextHook...');
		if (msg.text == null) return false;

		// 同じユーザーからの応答待ちリクエストが既に処理中の場合、二重に問い合わせない
		// (unsubscribeReplyはこの後すぐ行われるため、代わりにこのフラグで多重実行を防ぐ)
		if (this.processingUsers.has(msg.userId)) {
			this.log('Already processing a request for this user. Skipping.');
			return false;
		}

		let exist : AiChatHist | null = null;

		if (msg.isChat) {
			// チャット(DM)には会話概念が無く、コアのcontexts機構がsubscribeReply時のkey(=aichatHistのpostId)を
			// そのまま渡してくれるため、notes/conversationのようなAPI呼び出しは不要
			exist = this.aichatHist.findOne({
				postId: key
			});
			if (exist == null) {
				this.log('aichatHist entry is not found for this chat context.');
				return false;
			}
		} else {
			// msg.idをもとにnotes/conversationを呼び出し、該当のidかチェック
			const conversationData = await this.ai.api('notes/conversation', { noteId: msg.id });

			// 結果がnullやサイズ0の場合は終了
			if (conversationData == null || conversationData.length == 0 ) {
				this.log('conversationData is nothing.');
				return false;
			}

			// aichatHistに該当のポストが見つからない場合は終了
			for (const message of conversationData) {
				exist = this.aichatHist.findOne({
					postId: message.id
				});
				// 見つかった場合はそれを利用
				if (exist != null) break;
			}
			if (exist == null) {
				this.log('conversationData is not found.');
				return false;
			}
		}

		// 見つかった場合はunsubscribe&removeし、回答。今回のでsubscribe,insert,timeout設定
		this.log('unsubscribeReply & remove.');
		this.log(exist.type + ':' + exist.postId);
		if (exist.history) {
			for (const his of exist.history) {
				this.log(his.role + ':' + his.content);
			}
		}
		this.unsubscribeReply(key);
		this.aichatHist.remove(exist);

		// AIに問い合わせ
		this.processingUsers.add(msg.userId);
		let result: boolean;
		try {
			result = await this.handleAiChat(exist, msg);
		} finally {
			this.processingUsers.delete(msg.userId);
		}

		if (result) {
			return {
				reaction: 'like'
			};
		}
		return false;
	}

	@bindThis
	private async aichatRandomTalk() {
		this.log('AiChat(randomtalk) started');
		const tl = await this.ai.api('notes/local-timeline', {
			limit: 30
		});
		const interestedNotes = tl.filter(note =>
			note.userId !== this.ai.account.id &&
			note.text != null &&
			note.replyId == null &&
			note.renoteId == null &&
			note.cw == null &&
			note.files.length == 0 &&
			!note.user.isBot
		);

		// 対象が存在しない場合は処理終了
		if (interestedNotes == undefined || interestedNotes.length == 0) return false;

		// ランダムに選択
		const choseNote = interestedNotes[Math.floor(Math.random() * interestedNotes.length)];

		// aichatHistに該当のポストが見つかった場合は会話中のためaichatRandomTalkでは対応しない
		let exist : AiChatHist | null = null;

		// 選択されたノート自体が会話中のidかチェック
		exist = this.aichatHist.findOne({
			postId: choseNote.id
		});
		if (exist != null) return false;

		// msg.idをもとにnotes/childrenを呼び出し、会話中のidかチェック
		const childrenData = await this.ai.api('notes/children', { noteId: choseNote.id });
		if (childrenData != undefined) {
			for (const message of childrenData) {
				exist = this.aichatHist.findOne({
					postId: message.id
				});
				if (exist != null) return false;
			}
		}

		// msg.idをもとにnotes/conversationを呼び出し、会話中のidかチェック
		const conversationData = await this.ai.api('notes/conversation', { noteId: choseNote.id });
		if (conversationData != undefined) {
			for (const message of conversationData) {
				exist = this.aichatHist.findOne({
					postId: message.id
				});
				if (exist != null) return false;
			}
		}

		// 確率をクリアし、親愛度が指定以上、かつ、Botでない場合のみ実行
		if (Math.random() < this.randomTalkProbability) {
			this.log('AiChat(randomtalk) targeted: ' + choseNote.id);
		} else {
			this.log('AiChat(randomtalk) is end.');
			return false;
		}
		const friend: Friend | null = this.ai.lookupFriend(choseNote.userId);
		if (friend == null || friend.love < 7) {
			this.log('AiChat(randomtalk) end.Because there was not enough affection.');
			return false;
		} else if (choseNote.user.isBot) {
			this.log('AiChat(randomtalk) end.Because message author is bot.');
			return false;
		}

		const current : AiChatHist = {
			postId: choseNote.id,
			createdAt: Date.now(),// 適当なもの
			type: TYPE_OPENAI,		// 別のAPIをデフォルトにしてもよい
			fromMention: false,		// ランダムトークの場合はfalseとする
		};
		// AIに問い合わせ
		let targetedMessage = choseNote;
		if (choseNote.extractedText == undefined) {
			const data = await this.ai.api('notes/show', { noteId: choseNote.id });
			targetedMessage = new Message(this.ai, data);
		}
		const result = await this.handleAiChat(current, targetedMessage);

		if (result) {
			return {
				reaction: 'like'
			};
		}
		return false;
	}

	@bindThis
	private async handleAiChat(exist: AiChatHist, msg: Message) {
		let text: string | null, aiChat: AiChat;
		let prompt: string = '';
		if (config.prompt) {
			prompt = config.prompt;
		}
		// このJuice Serverのバージョンを伝えておく(聞かれたときに正しく答えられるように)
		const serverVersion = await this.getServerVersion();
		if (serverVersion != null) {
			prompt += 'また、このJuice Serverの現在のバージョンは' + serverVersion + 'です。バージョンを聞かれた場合はこの値をそのまま答えてください(自分でAPIを叩く必要があるかのような案内はしないこと)。';
		}
		const reName = RegExp(this.name, 'i');
		let reKigoType = RegExp(KIGO + exist.type, 'i');
		// openaiタイプは&openai/&chatgpt/&chatgpt4のいずれでも呼び出せるため、質問文からまとめて除去する
		if (exist.type === TYPE_OPENAI) {
			reKigoType = RegExp(`${KIGO}(${TYPE_OPENAI}|chatgpt4|chatgpt)`, 'i');
		}
		const extractedText = msg.extractedText;
		if (extractedText == undefined || extractedText.length == 0) return false;

		// Gemini API用にAPIのURLと置き換え用タイプを変更
		if (msg.includes([KIGO + GEMINI_FLASH])) {
			exist.api = GEMINI_20_FLASH_API;
			reKigoType = RegExp(KIGO + GEMINI_FLASH, 'i');
		} else if (msg.includes([KIGO + GEMINI_PRO])) {
			exist.api = GEMINI_15_PRO_API;
			reKigoType = RegExp(KIGO + GEMINI_PRO, 'i');
		}

		// groudingサポート
		if (msg.includes([GROUNDING_TARGET])) {
			exist.grounding = true;
		}
		// 設定で、デフォルトgroundingがONの場合、メンションから来たときは強制的にgroundingをONとする(ランダムトークの場合は勝手にGoogle検索するのちょっと気が引けるため...)
		if (exist.fromMention && config.aichatGroundingWithGoogleSearchAlwaysEnabled) {
			exist.grounding = true;
		}

		const friend: Friend | null = this.ai.lookupFriend(msg.userId);
		let friendName: string | undefined;
		if (friend != null && friend.name != null) {
			friendName = friend.name;
		} else if (msg.user.name) {
			friendName = msg.user.name;
		} else {
			friendName = msg.user.username;
		}

		const question = extractedText
							.replace(reName, '')
							.replace(reKigoType, '')
							.replace(GROUNDING_TARGET, '')
							.trim();

		// 質問内容にdocs.mk-juice.devの関連ページがあれば、内容をシステムプロンプトに追記する
		const docsContext = await buildDocsContext(question);
		if (docsContext) {
			prompt += '\n\n' + docsContext;
		}

		switch (exist.type) {
			case TYPE_GEMINI:
				// geminiの場合、APIキーが必須
				if (!config.geminiProApiKey) {
					msg.reply(serifs.aichat.nothing(exist.type));
					return false;
				}
				// チャット(DM)の添付ファイルは今のところ未対応(ノートの添付のみ画像として渡す)
				const noteFiles: base64File[] = msg.isChat ? [] : await this.note2base64File(msg.id);
				// 本文中のカスタム絵文字ショートコードを画像として解決し、添付ファイルに加える
				const geminiEmojis = await this.resolveEmojiImages(question);
				const base64Files: base64File[] = [...noteFiles, ...geminiEmojis.map(e => e.file)];
				aiChat = {
					question: question,
					prompt: prompt,
					api: GEMINI_20_FLASH_API,
					key: config.geminiProApiKey,
					history: exist.history,
					friendName: friendName,
					fromMention: exist.fromMention,
					emojiNames: geminiEmojis.map(e => e.name),
				};
				if (exist.api) {
					aiChat.api = exist.api;
				}
				if (exist.grounding) {
					aiChat.grounding = exist.grounding;
				}
				aiChat.prompt += this.describeSelfModel(aiChat.api.match(/\/models\/([^:/]+):/)?.[1] ?? aiChat.api, null);
				text = await this.genTextByGemini(aiChat, base64Files);
				break;

			case TYPE_PLAMO:
				// PLaMoの場合、APIキーが必須
				if (!config.pLaMoApiKey) {
					msg.reply(serifs.aichat.nothing(exist.type));
					return false;
				}
				aiChat = {
					question: msg.text,
					prompt: prompt,
					api: PLAMO_API,
					key: config.pLaMoApiKey,
					history: exist.history,
					friendName: friendName,
					fromMention: exist.fromMention
				};
				aiChat.prompt += this.describeSelfModel(PLAMO_MODEL, null);
				text = await this.genTextByPLaMo(aiChat);
				break;

			case TYPE_OPENAI:
				// OpenAI互換APIの場合、APIキーが必須
				if (!config.openaiApiKey) {
					msg.reply(serifs.aichat.nothing(exist.type));
					return false;
				}
				// チャット(DM)の添付ファイルは今のところ未対応(ノートの添付のみ画像として渡す)
				const openaiNoteFiles: base64File[] = msg.isChat ? [] : await this.note2base64File(msg.id);
				// 本文中のカスタム絵文字ショートコードを画像として解決し、添付ファイルに加える
				const openaiEmojis = await this.resolveEmojiImages(question);
				const openaiFiles: base64File[] = [...openaiNoteFiles, ...openaiEmojis.map(e => e.file)];
				aiChat = {
					question: question,
					prompt: prompt,
					api: config.openaiApiUrl ?? OPENAI_DEFAULT_API,
					key: config.openaiApiKey,
					history: exist.history,
					friendName: friendName,
					fromMention: exist.fromMention,
					emojiNames: openaiEmojis.map(e => e.name),
				};
				aiChat.prompt += this.describeSelfModel(config.openaiModel ?? OPENAI_DEFAULT_MODEL, config.openaiReasoningEffort ?? null);
				text = await this.genTextByOpenAI(aiChat, openaiFiles);
				break;

			case TYPE_CLAUDE:
				// Claudeの場合、APIキーが必須
				if (!config.claudeApiKey) {
					msg.reply(serifs.aichat.nothing(exist.type));
					return false;
				}
				// チャット(DM)の添付ファイルは今のところ未対応(ノートの添付のみ画像として渡す)
				const claudeNoteFiles: base64File[] = msg.isChat ? [] : await this.note2base64File(msg.id);
				// 本文中のカスタム絵文字ショートコードを画像として解決し、添付ファイルに加える
				const claudeEmojis = await this.resolveEmojiImages(question);
				const claudeFiles: base64File[] = [...claudeNoteFiles, ...claudeEmojis.map(e => e.file)];
				aiChat = {
					question: question,
					prompt: prompt,
					api: config.claudeApiUrl ?? CLAUDE_DEFAULT_API,
					key: config.claudeApiKey,
					history: exist.history,
					friendName: friendName,
					fromMention: exist.fromMention,
					emojiNames: claudeEmojis.map(e => e.name),
				};
				aiChat.prompt += this.describeSelfModel(config.claudeModel ?? CLAUDE_DEFAULT_MODEL, config.claudeEffort ?? CLAUDE_DEFAULT_EFFORT);
				text = await this.genTextByClaude(aiChat, claudeFiles);
				break;

			default:
				msg.reply(serifs.aichat.nothing(exist.type));
				return false;
		}

		if (text == null || text == '') {
			this.log('The result is invalid. It seems that tokens and other items need to be reviewed.')
			msg.reply(serifs.aichat.error(exist.type));
			return false;
		}

		this.log('Replying...');
		msg.reply(serifs.aichat.post(text, exist.type), { isAIGenerated: true }).then(reply => {
			// 履歴に登録
			if (!exist.history) {
				exist.history = [];
			}
			exist.history.push({ role: 'user', content: question });
			exist.history.push({ role: 'model', content: text });
			// 履歴が10件を超えた場合、古いものを削除
			if (exist.history.length > 10) {
				exist.history.shift();
			}
			this.aichatHist.insertOne({
				postId: reply.id,
				createdAt: Date.now(),
				type: exist.type,
				api: aiChat.api,
				history: exist.history,
				grounding: exist.grounding,
				fromMention: exist.fromMention,
			});

			this.log('Subscribe&Set Timer...');

			// メンション(またはチャット)をsubscribe
			this.subscribeReply(reply.id, msg.isChat, msg.isChat ? msg.userId : reply.id);

			// タイマーセット
			this.setTimeoutWithPersistence(TIMEOUT_TIME, {
				id: reply.id
			});
		});
		return true;
	}

	@bindThis
	private async timeoutCallback({id}) {
		this.log('timeoutCallback...');
		const exist = this.aichatHist.findOne({
			postId: id
		});
		this.unsubscribeReply(id);
		if (exist != null) {
			this.aichatHist.remove(exist);
		}
	}
}
