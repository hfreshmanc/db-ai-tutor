export interface Message {
  role: 'user' | 'model';
  parts: { text: string }[];
  attachment?: {
    name: string;
    content: string;
  };
}

export type ViewType = 'chat' | 'exercises' | 'design';

export interface NavItem {
  id: ViewType;
  label: string;
  icon: string;
}
