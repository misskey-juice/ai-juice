import { Client } from '@modelcontextprotocol/sdk/client/index.js';
import { StreamableHTTPClientTransport } from '@modelcontextprotocol/sdk/client/streamableHttp.js';
import log from '@/utils/log.js';
import config from '@/config.js';

// ローカル専用のced-mcp-server(loopbackのみ)
const LOCAL_MCP_SERVER_URL = 'http://127.0.0.1:10205/mcp';
// Kagi公式のMCPサーバー(kagiApiKeyを設定したときだけ使う)
const KAGI_MCP_SERVER_URL = 'https://mcp.kagi.com/mcp';
// Kagiを使うときは、ローカルのDuckDuckGo検索のツールを隠して検索をKagiに寄せる
const LOCAL_TOOLS_REPLACED_BY_KAGI = ['web_search', 'web_research'];

export type OpenAiFunctionTool = {
	type: 'function';
	function: {
		name: string;
		description?: string;
		parameters: unknown;
	};
};

type McpServer = {
	name: string;
	url: string;
	headers?: Record<string, string>;
	hiddenTools: string[];
	client: Client | null;
	connecting: Promise<Client> | null;
};

const servers: McpServer[] = [
	{
		name: 'local',
		url: LOCAL_MCP_SERVER_URL,
		hiddenTools: config.kagiApiKey ? LOCAL_TOOLS_REPLACED_BY_KAGI : [],
		client: null,
		connecting: null,
	},
	...(config.kagiApiKey ? [{
		name: 'kagi',
		url: KAGI_MCP_SERVER_URL,
		headers: { Authorization: `Bearer ${config.kagiApiKey}` },
		hiddenTools: [],
		client: null,
		connecting: null,
	}] : []),
];

// ツール名 → そのツールを持つサーバー(listOpenAiToolsで作り直す)
const toolServers = new Map<string, McpServer>();

// 接続は初回利用時に1回だけ行い、以降は使い回す(切断されていたら再接続する)
async function getClient(server: McpServer): Promise<Client> {
	if (server.client != null) return server.client;
	if (server.connecting != null) return server.connecting;

	server.connecting = (async () => {
		try {
			const c = new Client({ name: 'ai-juice', version: '1.0.0' });
			const transport = new StreamableHTTPClientTransport(new URL(server.url), {
				requestInit: server.headers ? { headers: server.headers } : undefined,
			});
			await c.connect(transport);
			server.client = c;
			return c;
		} finally {
			server.connecting = null;
		}
	})();

	return server.connecting;
}

// 接続エラー時は次回また接続し直せるように状態をリセットする
function resetClient(server: McpServer) {
	server.client = null;
	server.connecting = null;
}

function logError(message: string, err: unknown) {
	log(message);
	if (err instanceof Error) {
		log(`${err.name}\n${err.message}`);
	}
}

// 全MCPサーバーのツール一覧を、OpenAI互換APIのfunction calling形式に変換して返す
// (1つのサーバーにつながらなくても、ほかのサーバーのツールは使えるようにする)
export async function listOpenAiTools(): Promise<OpenAiFunctionTool[]> {
	const lists = await Promise.all(servers.map(async server => {
		try {
			const c = await getClient(server);
			const { tools } = await c.listTools();
			return tools
				.filter(tool => !server.hiddenTools.includes(tool.name))
				.map(tool => ({ server, tool }));
		} catch (err: unknown) {
			logError(`Error listing MCP tools (${server.name})`, err);
			resetClient(server);
			return [];
		}
	}));

	toolServers.clear();
	const result: OpenAiFunctionTool[] = [];
	for (const { server, tool } of lists.flat()) {
		// 同じ名前のツールがあれば、先に並んでいるサーバーのものを使う
		if (toolServers.has(tool.name)) continue;
		toolServers.set(tool.name, server);
		result.push({
			type: 'function' as const,
			function: {
				name: tool.name,
				description: tool.description,
				parameters: tool.inputSchema,
			},
		});
	}
	return result;
}

// MCPツールを1件呼び出し、結果をテキストにまとめて返す
export async function callMcpTool(name: string, args: Record<string, unknown>): Promise<string> {
	const server = toolServers.get(name);
	if (server == null) {
		return `ツールが見つかりません: ${name}`;
	}
	try {
		const c = await getClient(server);
		const result = await c.callTool({ name, arguments: args });
		const content = Array.isArray(result.content) ? result.content : [];
		const text = content
			.filter((item: any) => item?.type === 'text' && typeof item.text === 'string')
			.map((item: any) => item.text)
			.join('\n');
		if (text.length === 0) {
			return result.isError ? `ツール実行エラー: ${name}` : `(${name}から結果が返りませんでした)`;
		}
		return text;
	} catch (err: unknown) {
		logError(`Error calling MCP tool: ${name} (${server.name})`, err);
		resetClient(server);
		return `ツール呼び出しに失敗しました: ${name}`;
	}
}
